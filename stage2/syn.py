import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

class SynData(object):
    def __init__(self, dtype, ratio=0.8):
        self.ratio = ratio
        self.dtype = dtype
    
    def process(self, subgraphs):

        self.node_feat = subgraphs["node_feat"] # 原始圖node feature(原圖維度)
        self.node_feat_imp = subgraphs["node_feat_imp"] # k個子圖分別的node feature 重要度(原圖維度)
        self.edge_imp = subgraphs["edge_imp"] # k個子圖分別的edge 重要度(原圖維度)
        self.subgraphs_nodes = subgraphs["subgraphs_nodes"] # k個子圖的node
        self.subgraphs_edges = subgraphs["subgraphs_edges"] # k個子圖的edge
        self.edges_idx = subgraphs["subgraphs_edges_idx"] # k個子圖的edge index
        self.origraph_edges = subgraphs["origraph_edges"] # k個子圖的node在原圖中存在的所有邊

        self._sort_nodes()
        return self.pygdata



    # 針對會使用到的node 做重新排序
    def _sort_nodes(self):
        # 創建一個字典來儲存舊標籤到新標籤的映射
        node_mapping = {node.item(): i for i, node in enumerate(self.subgraphs_nodes)}

        # 使用映射來更新節點和邊的標籤
        self.subgraphs_nodes_sort = torch.tensor([node_mapping[node.item()] for node in self.subgraphs_nodes]) # 子圖的所有node
        self.subgraphs_edges_sort = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in self.subgraphs_edges.t()]).t() # 子圖的所有edge
        self.origraph_edges_sort = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in self.origraph_edges.t()]).t() # 在原圖內，與子圖node有連結的所有edge
        # 如果成功對上都沒有報錯，表示有被選上的edge，其節點對也都有被選到  ->成功
        
        # 在原圖內，但不在子圖集合的 edge
        set1 = set(map(tuple, self.subgraphs_edges_sort.t().numpy()))
        set2 = set(map(tuple, self.origraph_edges_sort.t().numpy()))
        self.not_in_subgraph = torch.tensor(list(set2 - set1)).t()
        
        # 將k個子圖的 node_feat_imp, edge_imp 重頭排序 (即移除0的列)
        self.node_feat_imp_sort = self.node_feat_imp[:, self.subgraphs_nodes, :]
        self.edge_imp_sort = self.edge_imp[:, self.edges_idx]

        # 排序後的 full node feature matrix
        self.node_feat_sort = self.node_feat[self.subgraphs_nodes]
        self._pyg_data()
    

    def _generate_edges_labels_wt_neg(self):
        if self.dtype == "train": 
            # 學習結構的邊暫時全放入所有子圖的邊
            # 希望每個epoch隨機sample學習結構的邊 & 計算loss
            # 訓練時，每個batch會再生成負樣本，這裡先沒有放入0
            # 計算loss的邊暫時先放入所有子圖的邊(全部1)
            edge_index = self.subgraphs_edges_sort
            edge_label = torch.ones((1, self.subgraphs_edges_sort.size(1)), dtype=torch.long)[0]
            edge_label_index = self.subgraphs_edges_sort
            edge_index_importance = self.edge_imp_sort 

        elif self.dtype == "valid": # valid 已加入固定的負樣本
            num_edges = int(self.subgraphs_edges_sort.shape[1] * self.ratio)

            # 選中於學習結構的邊
            selected_edges = random.sample(range(self.subgraphs_edges_sort.shape[1]), num_edges)
            edge_index = self.subgraphs_edges_sort[:, selected_edges]
            edge_index_importance = self.edge_imp_sort[:, selected_edges] # 學習結構的邊的重要度

            # 計算loss的邊
            remaining_edges = list(set(range(self.subgraphs_edges_sort.shape[1])) - set(selected_edges))
            edge_label_index = self.subgraphs_edges_sort[:, remaining_edges]

            # 給定圖的節點集合中隨機選擇兩個節點形成一條邊，並保證這條邊不出現在edge_index指定的已存在的邊集中，從而生成負樣本。
            self.negative_samples = negative_sampling(
                edge_index= self.origraph_edges_sort, num_nodes=len(self.subgraphs_nodes_sort),
                num_neg_samples=edge_label_index.shape[1], method='sparse') 
            
            negative_labels = torch.zeros((1, edge_label_index.shape[1]), dtype=torch.long)
            edge_label = torch.cat((torch.ones((1, edge_label_index.shape[1]), dtype=torch.long), negative_labels), dim=1)[0]
            edge_label_index = torch.cat((edge_label_index, self.negative_samples), dim=1)
    

        else: # test 已加入固定的負樣本
            edge_index = self.subgraphs_edges_sort
            edge_index_importance = self.edge_imp_sort

            self.negative_samples = negative_sampling(
                edge_index= self.origraph_edges_sort, num_nodes=len(self.subgraphs_nodes_sort),
                num_neg_samples=self.not_in_subgraph.shape[1], method='sparse') 

            negative_labels = torch.zeros((1, self.not_in_subgraph.shape[1]), dtype=torch.long)
            edge_label = torch.cat((torch.ones((1, self.not_in_subgraph.size(1)), dtype=torch.long), negative_labels), dim=1)[0]
            edge_label_index = torch.cat((self.not_in_subgraph, self.negative_samples), dim=1)

        return edge_index, edge_label, edge_label_index, edge_index_importance
    
    
    def _pyg_data(self):
        edge_index, edge_label, edge_label_index, edge_index_importance = self._generate_edges_labels_wt_neg()
        self.pygdata = Data(x=self.node_feat_sort, edge_index=edge_index,
                            edge_label=edge_label, edge_label_index=edge_label_index, 
                            node_feat_imp = self.node_feat_imp_sort, 
                            edge_imp = edge_index_importance, 
                            usefor = self.dtype) 
    


    
