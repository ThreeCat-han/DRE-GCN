

r"""
DREGCN
################################################
Reference:
    
"""

import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn

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
        self.device = config['device']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.tag_embedding = torch.nn.Embedding(num_embeddings=self.n_tags, embedding_dim=self.embedding_size)
        self.mf_loss = BPRLoss()
        self.tag_loss = MaskedBPRLoss()
        # self.reg_loss = EmbLoss()

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

        # 新增3start
        # 这里的to(self.device)全换成.cuda()试试{
        # self.user_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.user_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.item_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.pos_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.neg_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.hard_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.conf_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.easy_gate = nn.Linear(self.embedding_size, self.embedding_size).to(self.device)
        self.margin_model = nn.Linear(self.embedding_size, 1).to(self.device)
        # }
        # 新增end

    def get_norm_adj_mat(self, row_num, col_num, sp_inter):
        A = sp.dok_matrix((row_num + col_num, row_num + col_num), dtype=np.float32)
        inter_M = sp_inter
        if isinstance(inter_M, torch.Tensor):
            inter_M = sp.coo_matrix((inter_M.values().cpu().numpy(), inter_M.indices().cpu().numpy()),
                                    shape=(row_num, col_num))
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + row_num), inter_M.data))  # [1] * inter_M.nnz
        data_dict.update(dict(zip(zip(inter_M_t.row + row_num, inter_M_t.col), inter_M_t.data)))  # [1] * inter_M_t.nnz
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

    def split_ego_embeddings(self, row_num, col_num, emb):
        return torch.split(emb, [row_num, col_num])

    def forward(self, row_emb, col_emb, adj_mat, layers=2):
        all_embeddings = self.get_ego_embeddings(row_emb, col_emb)

        lightgcn_all_embeddings = all_embeddings + 0
        for _ in range(layers):
            all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)
            lightgcn_all_embeddings += all_embeddings
        lightgcn_all_embeddings = lightgcn_all_embeddings.div(1 + layers)
        ui_u_emb, ui_i_emb = self.split_ego_embeddings(row_emb.shape[0], col_emb.shape[0], lightgcn_all_embeddings)
        return ui_u_emb, ui_i_emb

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        if self.restore_user_ua is not None or self.restore_tag_ua is not None:
            self.restore_user_ua, self.restore_tag_ua = None, None
        if self.restore_item_ia is not None or self.restore_tag_ia is not None:
            self.restore_item_ia, self.restore_tag_ia = None, None

        ua_u_emb, ua_a_emb = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix,
                                          self.m_layers)
        user_ua = ua_u_emb[interaction[self.USER_ID]]
        tag_ua = torch.mm(user_ua, ua_a_emb.transpose(0, 1))
        pos_tag_ua = tag_ua.gather(1, interaction[self.TAG_ID])
        neg_tag_ua = tag_ua.gather(1, interaction[self.NEG_TAG_ID])

        ia_i_emb, ia_a_emb = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix,
                                          self.m_layers)
        item_ia = ia_i_emb[interaction[self.ITEM_ID]]
        tag_ia = torch.mm(item_ia, ia_a_emb.transpose(0, 1))
        pos_tag_ia = tag_ia.gather(1, interaction[self.TAG_ID])
        neg_tag_ia = tag_ia.gather(1, interaction[self.NEG_TAG_ID])

        mask = self.tag_mask(interaction[self.TAG_ID])

        tag_loss = self.tag_loss(pos_tag_ua + pos_tag_ia,
                                 neg_tag_ua + neg_tag_ia,
                                 mask)
        #原版
        # ui_u_emb, ui_i_emb = self.forward(self.user_embedding.weight + ua_u_emb,
        #                                   self.item_embedding.weight + ia_i_emb,
        #                                   self.ui_adj_matrix, self.n_layers)
        # user_ui = ui_u_emb[interaction[self.USER_ID]]
        # pos_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.ITEM_ID]])
        # neg_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.NEG_ITEM_ID]])

        # 新增start
        # ans
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # self.neg_seq_len=1
        neg_item_seq = neg_item.reshape((1, -1))# self.neg_seq_len=1
        neg_item_seq = neg_item_seq.T

        neg_item = neg_item_seq
        user_number = int(len(user) / 1)# self.neg_seq_len=1
        user = user[0:user_number]
        pos_item = pos_item[0:user_number]

        ui_u_emb, ui_i_emb = self.forward(self.user_embedding.weight + ua_u_emb,
                                                                self.item_embedding.weight + ia_i_emb,
                                                                self.ui_adj_matrix, self.n_layers)
        # user_all_embeddings, user_all_embeddings = self.forward()
        user_all_embeddings = ui_u_emb.unsqueeze(1).repeat(1, self.n_layers + 1, 1)
        item_all_embeddings = ui_i_emb.unsqueeze(1).repeat(1, self.n_layers + 1, 1)

        # print("item_all_embeddings的维度", item_all_embeddings.shape)
        # print("pos_item的维度", pos_item.shape)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        # print("neg_embeddings的维度", neg_embeddings.shape)

        s_e = u_embeddings
        p_e = pos_embeddings
        n_e = neg_embeddings
        # print("ne的维度", n_e.shape)
        batch_size = user.shape[0]

        gate_neg_hard = torch.sigmoid(self.item_gate(n_e) * self.user_gate(s_e).unsqueeze(1))
        n_hard = n_e * gate_neg_hard
        n_easy = n_e - n_hard

        p_hard = p_e.unsqueeze(1) * gate_neg_hard
        p_easy = p_e.unsqueeze(1) - p_hard

        import torch.nn.functional as F
        distance = torch.mean(F.pairwise_distance(n_hard, p_hard, p=2).squeeze(dim=1))
        temp = torch.norm(torch.mul(p_easy, n_easy), dim=-1)
        orth = torch.mean(torch.sum(temp, axis=-1))
        #测试部分start
        if torch.isnan(n_hard).any():
            raise ValueError("输入n_hard包含 NaN 值")
        if torch.isinf(n_hard).any():
            raise ValueError("输入n_hard包含无穷值")

        if torch.isnan(p_hard).any():
            raise ValueError("输入p_hard包含 NaN 值")
        if torch.isinf(p_hard).any():
            raise ValueError("输入p_hard包含无穷值")

        input_tensor = n_hard * p_hard

        # 通过 margin_model 获取输出
        margin_model_output = self.margin_model(input_tensor)

        # 检查 margin_model 的输出是否包含零或接近零的值
        if torch.isnan(margin_model_output).any():
            raise ValueError("输入margin_model_output包含 NaN 值")
        if torch.isinf(margin_model_output).any():
            raise ValueError("输入margin_model_output包含无穷值")
        #测试部分end
        epsilon = 1e-8
        safe_output = margin_model_output + epsilon
        margin = torch.sigmoid(1 / safe_output)
        # margin = torch.sigmoid(1 / self.margin_model(n_hard * p_hard))

        random_noise = torch.rand(n_easy.shape).to(self.device)
        magnitude = torch.nn.functional.normalize(random_noise, p=2, dim=-1) * margin * 0.1
        direction = torch.sign(p_easy - n_easy)
        noise = torch.mul(direction, magnitude)
        n_easy_syth = noise + n_easy
        n_e_ = n_hard + n_easy_syth
        hard_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_hard), axis=-1)  # [batch_size, K]
        easy_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_easy), axis=-1)  # [batch_size, K]
        syth_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e_), axis=-1)  # [batch_size, K]
        norm_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e), axis=-1)  # [batch_size, K]
        sns_loss = torch.mean(torch.log(1 + torch.exp(easy_scores - hard_scores).sum(dim=1)))
        dis_loss = distance + orth
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs]
        scores_false = syth_scores - norm_scores


        indices = torch.max(scores + 0.1 * scores_false, dim=1)[1].detach()#self.eps 暂定0.1

        # print("置换前 n_e_ 的形状:", n_e_.shape)

        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        neg_embeddings = neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1).squeeze(dim=1).sum(dim=-1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)


        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=False,
        )
        loss = mf_loss + 1e-05 * reg_loss + 0.15 * (sns_loss + dis_loss)
               # + self.tag_weight * tag_loss#gamma也暂定0.1,reg_weight: 1e-05

        # loss = mf_loss + 0.2 * (sns_loss + dis_loss) + + self.tag_weight * tag_loss  # gamma也暂定0.1
        #新增end

        # mf_loss = self.mf_loss(pos_item_ui.sum(dim=-1), neg_item_ui.sum(dim=-1))#原

        # loss = mf_loss + self.tag_weight * tag_loss#原
        return loss

    # regloss
    def reg_loss(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=2
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= 2
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=2)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

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
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight,
                                                                     self.tag_embedding.weight, self.ua_adj_matrix,
                                                                     self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight,
                                                                     self.tag_embedding.weight, self.ia_adj_matrix,
                                                                     self.m_layers)

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.user_embedding.weight + self.restore_user_ua,
                                                                    self.item_embedding.weight + self.restore_item_ia,
                                                                    self.ui_adj_matrix, self.n_layers)

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

    def tag_predict(self, interaction):
        if self.restore_user_ua is None or self.restore_tag_ua is None:
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight,
                                                                     self.tag_embedding.weight, self.ua_adj_matrix,
                                                                     self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight,
                                                                     self.tag_embedding.weight, self.ia_adj_matrix,
                                                                     self.m_layers)

        scores = torch.matmul(self.restore_user_ua[interaction[self.USER_ID]], self.restore_tag_ua.transpose(0, 1)) + \
                 torch.matmul(self.restore_item_ia[interaction[self.ITEM_ID]], self.restore_tag_ia.transpose(0, 1))

        return scores