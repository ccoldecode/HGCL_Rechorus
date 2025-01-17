import torch as t
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from models.BaseModel import GeneralModel

class HGCL(GeneralModel):
    reader = 'HGCLReader'
    runner = 'BUIRRunner'

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        # 从corpus获取用户和物品数量
        self.n_users = corpus.n_users
        self.n_items = corpus.n_items
        
        # 模型参数
        self.hide_dim = args.hidden_size
        self.n_layers = args.n_layers
        self.tau = args.tau
        self.lambda_cl = args.lambda_cl
        
        # BUIR相关参数
        self.momentum = args.momentum  # 添加动量参数

        # 获取异构图结构
        self.uuMat = corpus.user_user_mat
        self.iiMat = corpus.item_item_mat
        self.uiMat = corpus.user_item_mat
        
        self._define_params()
        self.apply(self.init_weights)

        # 初始化目标网络
        self._init_target()

    def _init_target(self):
        """初始化目标网络"""
        # 目标网络的用户嵌入
        self.target_user_embedding = nn.Embedding(self.n_users, self.hide_dim)
        self.target_item_embedding = nn.Embedding(self.n_items, self.hide_dim)
        
        # 目标网络的GCN层
        self.target_encoder = nn.ModuleList([
            GCN_layer() for _ in range(self.n_layers)
        ])
        
        # 初始化目标网络的参数
        for param_o, param_t in zip(self.user_embedding.parameters(), self.target_user_embedding.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
            
        for param_o, param_t in zip(self.item_embedding.parameters(), self.target_item_embedding.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
            
        for encoder_o, encoder_t in zip(self.encoder, self.target_encoder):
            for param_o, param_t in zip(encoder_o.parameters(), encoder_t.parameters()):
                param_t.data.copy_(param_o.data)
                param_t.requires_grad = False

    def _update_target(self):
        """更新目标网络"""
        for param_o, param_t in zip(self.user_embedding.parameters(), self.target_user_embedding.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
            
        for param_o, param_t in zip(self.item_embedding.parameters(), self.target_item_embedding.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
            
        for encoder_o, encoder_t in zip(self.encoder, self.target_encoder):
            for param_o, param_t in zip(encoder_o.parameters(), encoder_t.parameters()):
                param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def _define_params(self):
        # 初始化嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.hide_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.hide_dim)
        
        # GCN层
        self.encoder = nn.ModuleList([
            GCN_layer() for _ in range(self.n_layers)
        ])
        
        # 门控机制参数
        self.gating_weight_u = nn.Parameter(t.FloatTensor(self.hide_dim, self.hide_dim))
        self.gating_weight_i = nn.Parameter(t.FloatTensor(self.hide_dim, self.hide_dim))
        nn.init.xavier_uniform_(self.gating_weight_u)
        nn.init.xavier_uniform_(self.gating_weight_i)

        # 特征掩码参数（用于对比学习）
        # 使用 sigmoid 确保值在 [0,1] 范围内
        self.feature_mask = nn.Parameter(t.zeros(self.hide_dim))
        self.dropout = nn.Dropout(p=0.1)

    def _get_cl_views(self, embeddings):
        # 生成对比学习视图
        # 第一个视图使用dropout
        view1 = self.dropout(embeddings)
        
        # 第二个视图使用特征掩码
        # 使用 sigmoid 确保概率值在 [0,1] 范围内
        mask = t.sigmoid(self.feature_mask)
        view2 = embeddings * t.bernoulli(mask).to(embeddings.device)
        
        return view1, view2

    def loss(self, out_dict):
        # 计算BPR损失
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        bpr_loss = -(pos_pred.unsqueeze(1) - neg_pred).sigmoid().log().mean()
        
        # 计算对比损失
        u_cl_loss = self._compute_cl_loss(out_dict['user_view1'], out_dict['user_view2'])
        i_cl_loss = self._compute_cl_loss(out_dict['item_view1'], out_dict['item_view2'])
        cl_loss = (u_cl_loss + i_cl_loss) / 2
        
        # 总损失
        loss = bpr_loss + self.lambda_cl * cl_loss
        return loss

    def _compute_cl_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos_score = (view1 * view2).sum(dim=-1)
        neg_score = view1 @ view2.t()
        pos_score = t.exp(pos_score / self.tau)
        neg_score = t.exp(neg_score / self.tau).sum(dim=-1)
        return -t.log(pos_score / neg_score).mean()

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', type=int, default=64,
                          help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                          help='Number of GCN layers.')
        parser.add_argument('--tau', type=float, default=0.2,
                          help='Temperature parameter.')
        parser.add_argument('--lambda_cl', type=float, default=0.1,
                          help='Weight for contrastive loss.')
        parser.add_argument('--momentum', type=float, default=0.995,
                          help='Momentum for target network update.')  # 添加动量参数
        return GeneralModel.parse_model_args(parser)

    def _gcn_encode(self, embeddings, adj_mat, feed_dict):
        """
        Args:
            embeddings: [batch_size * num_items, hidden_dim] 物品嵌入
            adj_mat: [n_nodes, n_nodes] 邻接矩阵
            feed_dict: 数据字典
        """
        # 1. 准备完整的嵌入矩阵
        full_embeddings = t.zeros(adj_mat.shape[0], self.hide_dim).to(embeddings.device)
        
        # 2. 获取批次索引和形状信息
        if 'user_id' in feed_dict:
            # 处理用户嵌入
            batch_indices = feed_dict['user_id'].long()
            full_embeddings[batch_indices] = embeddings
        else:
            # 处理物品嵌入
            item_ids = feed_dict['item_id'].view(-1).long()  # [batch_size * num_items]
            valid_indices = item_ids[item_ids < adj_mat.shape[0]]
            full_embeddings[valid_indices] = embeddings[item_ids < adj_mat.shape[0]]

        # 3. 图卷积传播
        all_embeddings = [full_embeddings]
        for layer in self.encoder:
            full_embeddings = layer(full_embeddings, adj_mat)
            all_embeddings.append(full_embeddings)
        
        # 4. 取出批处理对应的嵌入
        final_embeddings = t.stack(all_embeddings, dim=1).mean(dim=1)
        
        if 'user_id' in feed_dict:
            # 返回用户嵌入
            valid_indices = feed_dict['user_id'][feed_dict['user_id'] < final_embeddings.size(0)]
            return final_embeddings[valid_indices.long()]
        else:
            # 返回物品嵌入
            valid_item_ids = item_ids[item_ids < final_embeddings.size(0)]
            return final_embeddings[valid_item_ids]

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id']  # [256]
        item_ids = feed_dict['item_id']  # [256, 100]
        
        # 1. 基础嵌入
        user_embed = self.user_embedding(user_ids)  # [256, 64]
        item_embed = self.item_embedding(item_ids.view(-1))  # [25600, 64]
        
        # 2. 图卷积编码
        user_gcn = self._gcn_encode(user_embed, self.uuMat, feed_dict)  # [256, 64]
        item_gcn = self._gcn_encode(item_embed, self.iiMat, {'item_id': item_ids.view(-1)})  # [25600, 64]
        
        # 3. 生成对比视图
        user_view1, user_view2 = self._get_cl_views(user_gcn)
        item_view1, item_view2 = self._get_cl_views(item_gcn)
        
        # 4. 预测分数
        item_gcn = item_gcn.view(user_ids.size(0), -1, self.hide_dim)  # [256, 100, 64]
        prediction = t.sum(user_gcn.unsqueeze(1) * item_gcn, dim=-1)  # [256, 100]
        
        out_dict = {
            'prediction': prediction,
            'user_view1': user_view1,
            'user_view2': user_view2,
            'item_view1': item_view1,
            'item_view2': item_view2
        }
        return out_dict

class GCN_layer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, features, adj_mat):
        """
        Args:
            features: [n_nodes, hidden_dim] 节点特征矩阵
            adj_mat: [n_nodes, n_nodes] 邻接矩阵
        """
        # 标准化邻接矩阵
        if isinstance(adj_mat, sp.spmatrix):
            adj_mat = self._normalize_adj(adj_mat)
            adj_mat = self._sparse_mx_to_torch_sparse_tensor(adj_mat).to(features.device)
        
        # 图卷积操作
        return t.sparse.mm(adj_mat, features)  # 使用sparse.mm替代spmm

    def _normalize_adj(self, adj):
        """对称归一化邻接矩阵"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """将scipy稀疏矩阵转换为torch稀疏张量"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)