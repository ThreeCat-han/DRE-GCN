

r"""
DREGCN
################################################
Reference:
    
"""

import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.init import xavier_uniform_initialization

from Custom.loss import MaskedBPRLoss
from Custom.recommender import TagSampleRecommender

class DREGCN(TagSampleRecommender):
    r"""
    DREGCN is a model for joint task of item recommendation and explanation ranking.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DREGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.m_layers = config['m_layers']
        self.tag_weight = config['tag_weight']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.tag_embedding = torch.nn.Embedding(num_embeddings=self.n_tags, embedding_dim=self.embedding_size)
        self.mf_loss = BPRLoss()
        self.tag_loss = MaskedBPRLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_user_ua = None
        self.restore_tag_ua = None
        self.restore_item_ia = None
        self.restore_tag_ia = None

        # generate intermediate data
        self.ui_adj_matrix = self.get_norm_adj_mat(self.n_users, self.n_items, self.interaction_matrix).to(self.device)
        self.ua_adj_matrix = self.get_norm_adj_mat(self.n_users, self.n_tags, self.user_score).to(self.device)
        self.ia_adj_matrix = self.get_norm_adj_mat(self.n_items, self.n_tags, self.item_score).to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e',
                                     'restore_user_ua', 'restore_tag_ua',
                                     'restore_item_ia', 'restore_tag_ia']

    def get_norm_adj_mat(self, row_num, col_num, sp_inter):
        A = sp.dok_matrix((row_num + col_num, row_num + col_num), dtype=np.float32)
        inter_M = sp_inter
        if isinstance(inter_M, torch.Tensor):
            inter_M = sp.coo_matrix((inter_M.values().cpu().numpy(), inter_M.indices().cpu().numpy()), shape=(row_num, col_num))
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + row_num), inter_M.data)) #  [1] * inter_M.nnz
        data_dict.update(dict(zip(zip(inter_M_t.row + row_num, inter_M_t.col), inter_M_t.data))) #  [1] * inter_M_t.nnz
        A._update(data_dict)
        sumArr = A.sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        D = sp.diags(np.power(diag, -0.5))
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, row_emb, col_emb):
        return torch.cat([row_emb, col_emb], dim=0)

    # 新增DLN损失：
    def cal_user_neicl(self, user, user_all_embeddings, item_all_embeddings):
        u_norm = F.normalize(user_all_embeddings[user], dim=1)
        i_norm = F.normalize(item_all_embeddings, dim=1)
        self.uiA_dict = self.get_uiA_dict()
        scores = torch.matmul(u_norm, i_norm.transpose(0, 1))
        u_list = []
        i_list = []
        weight_list = []
        uidx_list = []
        sparse_shape = (len(user), self.n_items)
        i = 0
        for e in user:
            u_list.extend([i] * len(self.uiA_dict[e]['neighbors']))
            i_list.extend(self.uiA_dict[e]['neighbors'])
            weight_list.extend(self.uiA_dict[e]['neighbors_value'])
            uidx_list.extend([e] * len(self.uiA_dict[e]['neighbors']))
            i += 1
        weight = torch.tensor(weight_list).to(self.device)
        u_array = np.array(u_list)
        i_array = np.array(i_list) - self.n_users
        pos_indices = torch.tensor(np.vstack((u_array, i_array)))

        pos_scores_tau1 = torch.exp(scores[[u_array, i_array]] / self.tau)
        pos_sparse_tau1 = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores_tau1, size=sparse_shape)
        pos_sum_tau1 = torch.sparse.sum(pos_sparse_tau1, dim=1).to_dense()

        pos_scores = torch.exp(scores[[u_array, i_array]] / self.tau)
        # pos_scores = pos_scores * weight
        pos_sparse = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores, size=sparse_shape)
        pos_sum = torch.sparse.sum(pos_sparse, dim=1).to_dense()

        all_sum = torch.sum(torch.exp(scores / self.tau), dim=1)
        neg_sum = (all_sum - pos_sum)
        gama = 1e-10
        L = -torch.mean(torch.log(gama + pos_sum_tau1 / neg_sum))
        return L

    def cal_item_neicl(self, item, user_all_embeddings, item_all_embeddings):
        u_norm = F.normalize(user_all_embeddings, dim=1)
        i_norm = F.normalize(item_all_embeddings[item], dim=1)
        scores = torch.matmul(i_norm, u_norm.transpose(0, 1))
        item_gID = item + self.n_users
        u_list = []
        i_list = []
        weight_list = []
        iidx_list = []
        sparse_shape = (len(item), self.n_users)
        idx = 0
        for e in item_gID:
            i_list.extend([idx] * len(self.uiA_dict[e]['neighbors']))
            u_list.extend(self.uiA_dict[e]['neighbors'])
            weight_list.extend(self.uiA_dict[e]['neighbors_value'])
            iidx_list.extend([e] * len(self.uiA_dict[e]['neighbors']))
            idx += 1
        weight = torch.tensor(weight_list).to(self.device)
        i_array = np.array(i_list)
        u_array = np.array(u_list)
        pos_indices = torch.tensor(np.vstack((i_array, u_array)))

        pos_scores_tau1 = torch.exp(scores[[i_array, u_array]] / self.tau)
        pos_sparse_tau1 = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores_tau1, size=sparse_shape)
        pos_sum_tau1 = torch.sparse.sum(pos_sparse_tau1, dim=1).to_dense()

        pos_scores = torch.exp(scores[[i_array, u_array]] / self.tau)
        pos_sparse = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores, size=sparse_shape)
        pos_sum = torch.sparse.sum(pos_sparse, dim=1).to_dense()

        all_sum = torch.sum(torch.exp(scores / self.tau), dim=1)
        neg_sum = (all_sum - pos_sum)
        gama = 1e-10
        L = -torch.mean(torch.log(gama + pos_sum_tau1 / neg_sum))
        return L

    # 新增
    def get_uiA_dict(self):
        node0 = self.ui_adj_matrix.coalesce().indices()[0].cpu().tolist()
        node1 = self.ui_adj_matrix.coalesce().indices()[1].cpu().tolist()
        values = self.ui_adj_matrix.coalesce().values().cpu().tolist()
        uiA_dict = dict()
        for idx in range(len(node0)):
            if node0[idx] not in uiA_dict:
                neighbors = node1[idx]
                neighbors_value = values[idx]
                uiA_dict.update({node0[idx]: {'neighbors': [neighbors], 'neighbors_value': [neighbors_value]}})
            else:
                uiA_dict[node0[idx]]['neighbors'].append(node1[idx])
                uiA_dict[node0[idx]]['neighbors_value'].append(values[idx])
        return uiA_dict

    def split_ego_embeddings(self, row_num, col_num, emb):
        return torch.split(emb, [row_num, col_num])

    # def forward(self, row_emb, col_emb, adj_mat, layers=2):
    #     all_embeddings = self.get_ego_embeddings(row_emb, col_emb)
    #
    #     lightgcn_all_embeddings = all_embeddings + 0
    #     for _ in range(layers):
    #         # all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)
    #         # {改动1
    #         side_new = torch.sparse.mm(adj_mat, all_embeddings)
    #         if _ > 2:  # 对UAIA的嵌入更换lr（考虑到它们不同的过度平滑敏感性）
    #             all_embeddings = side_new + all_embeddings
    #         else:
    #             all_embeddings = side_new
    #             #}
    #         lightgcn_all_embeddings += all_embeddings
    #     lightgcn_all_embeddings = lightgcn_all_embeddings.div(1 + layers)
    #     ui_u_emb, ui_i_emb = self.split_ego_embeddings(row_emb.shape[0], col_emb.shape[0], lightgcn_all_embeddings)
    #     return ui_u_emb, ui_i_emb

    #改
    def forward(self, row_emb, col_emb, adj_mat, layers=2, flag=False):
        # tag_emb = self.tag_embedding.weight
        # user_emb = self.user_embedding.weight
        # item_emb = self.item_embedding.weight
        user_emb = row_emb
        item_emb = col_emb#14 Jun 11:42    INFO  recall@10   : 0.049803
        # all_embeddings = self.get_ego_embeddings(row_emb, col_emb)
        #
        # lightgcn_all_embeddings = all_embeddings + 0
        # for _ in range(layers):
        #     all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)
        #     # #{改动1
        #     # side_new = torch.sparse.mm(adj_mat, all_embeddings)
        #     # if _ > 2:  # 对UAIA的嵌入更换lr（考虑到它们不同的过度平滑敏感性）
        #     #     all_embeddings = side_new + all_embeddings
        #     # else:
        #     #     all_embeddings = side_new
        #     # #}
        #     lightgcn_all_embeddings += all_embeddings
        # lightgcn_all_embeddings = lightgcn_all_embeddings.div(1 + layers)
        # # ui_u_emb, ui_i_emb = self.split_ego_embeddings(row_emb.shape[0], col_emb.shape[0], lightgcn_all_embeddings)

        # {改动2
        if flag == False:#ua和ia的处理
            all_embeddings = self.get_ego_embeddings(row_emb, col_emb)
            lightgcn_all_embeddings = all_embeddings + 0
            for _ in range(layers):
                #
                # all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)#baocuo
                side_embeddings = torch.sparse.mm(adj_mat, all_embeddings)  # baocuo
                #新增残差结构：
                # if _>2:#yelp最优
                # if _ > 1:#Trip最优、hotel最优
                # if _ > 2:  #amazon第一版改进
                if _ > 1:  # amazon第2版改进
                    all_embeddings = side_embeddings + all_embeddings
                else:
                    all_embeddings = side_embeddings
                lightgcn_all_embeddings += all_embeddings
        else:#用于推荐的ui处理

            all_embeddings = self.get_ego_embeddings(user_emb, item_emb)
            ua_u_emb = user_emb - self.user_embedding.weight
            ia_i_emb = item_emb - self.item_embedding.weight

            # tag_emb_padded_ua = self.get_ego_embeddings(user_emb, tag_emb)
            #
            # usertag_embeddings, taguser_embeddings = self.split_ego_embeddings(user_emb.shape[0], tag_emb.shape[0],
            #                                                              tag_emb_padded_ua)
            #
            # tag_emb_padded_ia = self.get_ego_embeddings(item_emb, tag_emb)
            # #
            # # itemtag_embeddings, tagitem_embeddings = self.split_ego_embeddings(item_emb.shape[0], tag_emb.shape[0],
            # #                                                              tag_emb_padded_ia)
            lightgcn_all_embeddings = all_embeddings + 0
            for _ in range(layers):
                # all_embeddings = torch.sparse.mm(self.ui_adj_matrix, all_embeddings)#原
                side_embeddings = torch.sparse.mm(self.ui_adj_matrix, all_embeddings)
                # 分离 user and item embeddings
                user_embeddings, item_embeddings = self.split_ego_embeddings(user_emb.shape[0], item_emb.shape[0], side_embeddings)

                weights1 = (F.cosine_similarity(ua_u_emb, user_embeddings, dim=-1)) * 0.1 + 1
                # weights1 = (F.cosine_similarity(usertag_embeddings, user_embeddings, dim=-1)) * 0.1 + 1
                user_embeddings = torch.einsum('a,ab->ab', weights1, user_embeddings)

                weights2 = (F.cosine_similarity(ia_i_emb, item_embeddings, dim=-1)) * 0.1 + 1
                # weights2 = (F.cosine_similarity(itemtag_embeddings, item_embeddings, dim=-1)) * 0.1 + 1
                item_embeddings = torch.einsum('a,ab->ab', weights2, item_embeddings)

                #重新组合成新的side_embeddings
                side_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
                # print("8的维度", side_embeddings.shape)

                #新增残差结构：
                # if _>1:#yelp最优
                # if _> 1:#Trip最优、hotel最优
                if _ > 3:  #amazon第一版改进
                # if _ > 2:  # Trip试一下(不行)
                    all_embeddings = side_embeddings + all_embeddings
                else:
                    all_embeddings = side_embeddings

                # all_embeddings = side_embeddings#
                lightgcn_all_embeddings += all_embeddings
        # }
        lightgcn_all_embeddings = lightgcn_all_embeddings.div(1 + layers)
        # print("9的维度", lightgcn_all_embeddings.shape)
        if flag == False:#ua和ia的处理
            ui_u_emb, ui_i_emb = self.split_ego_embeddings(row_emb.shape[0], col_emb.shape[0], lightgcn_all_embeddings)
        else:
            ui_u_emb, ui_i_emb = self.split_ego_embeddings(user_emb.shape[0], item_emb.shape[0], lightgcn_all_embeddings)
        return ui_u_emb, ui_i_emb

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        if self.restore_user_ua is not None or self.restore_tag_ua is not None:
            self.restore_user_ua, self.restore_tag_ua = None, None
        if self.restore_item_ia is not None or self.restore_tag_ia is not None:
            self.restore_item_ia, self.restore_tag_ia = None, None

        ua_u_emb, ua_a_emb = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        user_ua = ua_u_emb[interaction[self.USER_ID]]
        tag_ua = torch.mm(user_ua, ua_a_emb.transpose(0, 1))
        pos_tag_ua = tag_ua.gather(1, interaction[self.TAG_ID])
        neg_tag_ua = tag_ua.gather(1, interaction[self.NEG_TAG_ID])

        ia_i_emb, ia_a_emb = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)
        item_ia = ia_i_emb[interaction[self.ITEM_ID]]
        tag_ia = torch.mm(item_ia, ia_a_emb.transpose(0, 1))
        pos_tag_ia = tag_ia.gather(1, interaction[self.TAG_ID])
        neg_tag_ia = tag_ia.gather(1, interaction[self.NEG_TAG_ID])

        mask = self.tag_mask(interaction[self.TAG_ID])
        
        tag_loss = self.tag_loss(pos_tag_ua + pos_tag_ia,
                                 neg_tag_ua + neg_tag_ia,
                                 mask)

        ui_u_emb, ui_i_emb = self.forward(self.user_embedding.weight + ua_u_emb,
                                          self.item_embedding.weight + ia_i_emb,
                                          self.ui_adj_matrix, self.n_layers,True)
        user_ui = ui_u_emb[interaction[self.USER_ID]]
        pos_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.ITEM_ID]])
        neg_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.NEG_ITEM_ID]])

        # mf_loss = self.mf_loss(pos_item_ui.sum(dim=-1), neg_item_ui.sum(dim=-1))#原
        #新增
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        loss_list = []

        if len(user) > 0:
            user_neicl_loss = self.cal_user_neicl(user, ui_u_emb, ui_i_emb)
            loss_list.append(user_neicl_loss)
        if len(item) > 0:
            item_neicl_loss = self.cal_item_neicl(item, user_all_embeddings, item_all_embeddings)
            loss_list.append(item_neicl_loss)

        final_loss = 0
        for loss in loss_list:
            final_loss = final_loss + loss
        mf_loss = final_loss#改end

        loss = mf_loss + self.tag_weight * tag_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_ua is None or self.restore_tag_ua is None:
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.user_embedding.weight + self.restore_user_ua,
                                                                    self.item_embedding.weight + self.restore_item_ia,
                                                                    self.ui_adj_matrix, self.n_layers,True)

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

    def tag_predict(self, interaction):
        if self.restore_user_ua is None or self.restore_tag_ua is None:
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)
        
        scores = torch.matmul(self.restore_user_ua[interaction[self.USER_ID]], self.restore_tag_ua.transpose(0, 1)) + \
                 torch.matmul(self.restore_item_ia[interaction[self.ITEM_ID]], self.restore_tag_ia.transpose(0, 1))

        return scores