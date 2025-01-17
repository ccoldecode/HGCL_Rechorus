from helpers.KGReader import KGReader
import scipy.sparse as sp
import numpy as np
import logging
from collections import defaultdict

class HGCLReader(KGReader):
    def __init__(self, args):
        super().__init__(args)  # 先调用父类初始化
        
        # 确保基础数据已经处理完成
        self._prepare_user_history()
        self._construct_heterogeneous_graph()

    def _prepare_user_history(self):
        """准备用户历史交互数据"""
        self.user_his = defaultdict(list)
        for uid, iid, _ in zip(self.data_df['train']['user_id'], 
                              self.data_df['train']['item_id'],
                              self.data_df['train']['time']):
            self.user_his[uid].append((iid, 1.0))  # 使用简单的交互权重1.0

    def _construct_heterogeneous_graph(self):
        """构建异构图的邻接矩阵"""
        logging.info('Constructing heterogeneous graph...')
        
        # 1. 用户-用户交互图
        self.user_user_mat = self._build_user_user_graph()
        
        # 2. 物品-物品关系图
        self.item_item_mat = self._build_item_item_graph()
        
        # 3. 用户-物品交互图
        self.user_item_mat = self._build_user_item_graph()
        
        logging.info(f'[Graph] #U-U edges: {self.user_user_mat.nnz}')
        logging.info(f'[Graph] #I-I edges: {self.item_item_mat.nnz}')
        logging.info(f'[Graph] #U-I edges: {self.user_item_mat.nnz}')

    def _build_user_user_graph(self):
        """基于共同交互构建用户-用户图"""
        user_item_mat = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        # 构建用户-物品交互矩阵
        for u in self.user_his:
            for item, _ in self.user_his[u]:
                user_item_mat[u, item] = 1.0
                
        # 计算用户-用户相似度
        user_user_mat = user_item_mat @ user_item_mat.T
        user_user_mat.setdiag(0)  # 移除自环
        return user_user_mat.tocsr()

    def _build_item_item_graph(self):
        """基于知识图谱关系构建物品-物品图"""
        item_item_mat = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        
        # 1. 从知识图谱中提取物品关系
        if hasattr(self, 'triplet_set'):
            for h, r, t in self.triplet_set:
                if h < self.n_items and t < self.n_items:
                    item_item_mat[h, t] = 1.0
                    item_item_mat[t, h] = 1.0  # 对称化
        
        # 2. 基于共现构建物品关系（备选方案）
        if item_item_mat.nnz == 0:
            user_item_mat = self._build_user_item_graph()
            item_item_mat = user_item_mat.T @ user_item_mat
            item_item_mat.setdiag(0)
            
        return item_item_mat.tocsr()

    def _build_user_item_graph(self):
        """构建用户-物品交互图"""
        user_item_mat = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        # 添加历史交互
        for u in self.user_his:
            for item, _ in self.user_his[u]:
                user_item_mat[u, item] = 1.0
                
        return user_item_mat.tocsr()

    @staticmethod
    def parse_data_args(parser):
        parser = KGReader.parse_data_args(parser)
        parser.add_argument('--build_ii_from_kg', type=int, default=1,
                          help='Whether to build item-item graph from knowledge graph')
        return parser