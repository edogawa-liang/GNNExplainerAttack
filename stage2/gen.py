# Generating Explanatory Subgraphs and New Dataset Synthesis
import torch
import pickle
import random
from importlib import reload
import stage1_model
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer



class GeneratingSubgraphs(object):
    '''
    GNNExplainer生成子圖，並確認k個子圖有路徑相連
    '''
    def __init__(self, use_data, k, epoch):
        self.use_data = use_data
        self.k = k
        self.epoch = epoch
        self._load_data()
        self._explainer_setting()

    def _load_data(self):
        dataset_dir = f'dataset/{self.use_data}'
        with open(f'{dataset_dir}/{self.use_data}.pickle', 'rb') as file:
            self.data = pickle.load(file)


    def _explainer_setting(self):
        # Reload model and determine hop
        reload(stage1_model)
        hop = 3 if self.use_data == "Facebook" else 2
        GNNmodel = stage1_model.FBGCN() if self.use_data == "Facebook" else stage1_model.GitGCN()

        self.explainer = Explainer(
            model = GNNmodel,
            algorithm=GNNExplainer(epochs=self.epoch, num_hops=hop),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
                ),
        )


    def generate_or_load_subgraphs(self, use_existing, explanations=None):
        self.explanations = explanations
        if use_existing == True:
            self.node_id_explain = list(self.explanations.keys())
            self._subgraphInfo()
            self._summary()
            return self.subgraphs 

        else:
            # generate k subgraphs
            self._generate_subgraphs()
            self._subgraphInfo()
            self._check_path()     
            self._summary()
            return self.explanations, self.subgraphs 
        

    def _generate_subgraphs(self):
        self.node_id_explain = random.sample(range(self.data.x.shape[0]), self.k)
        self.explanations = {}
        for n in self.node_id_explain:
            explanation = self.explainer(self.data.x, self.data.edge_index, index=n)
            self.explanations[n] = explanation


    # 找出所有子圖的node和edge，以及 node feature, edge的重要度 
    def _subgraphInfo(self):
        all_features = []
        all_edges_importance = []

        for id_explain in self.node_id_explain:
            explain = self.explanations[id_explain]
            
            # node feature 重要度
            all_features.append(explain.node_mask)
            # edge 重要度
            all_edges_importance.append(explain.edge_mask)

        # 將子圖中的 node feature 合併在一起 
        self.features_importance = torch.stack(all_features)
        nodes_idx = (torch.max(self.features_importance, dim=0)[0] != 0).any(dim=1).nonzero().squeeze() #原圖被挑入子圖的node index

        # 將子圖中的 edge 重要度 合併在一起
        self.edges_importance = torch.stack(all_edges_importance)
        self.edges_idx = torch.max(self.edges_importance, dim=0)[0].nonzero().squeeze()# 原圖被挑入子圖的「第幾個」edge index
        
        # 子圖的node和edge index
        self.subgraphs_nodes = nodes_idx
        self.subgraphs_edges = self.data.edge_index[:, self.edges_idx]

        self._find_edge_in_origraph()


    # 挑選出在原圖中，與 子圖節點集合中相連的所有邊
    def _find_edge_in_origraph(self):
        edges = self.data.edge_index.t().tolist()
        selected_edges = [edge for edge in edges if edge[0] in self.subgraphs_nodes and edge[1] in self.subgraphs_nodes]
        self.origraph_edges = torch.tensor(selected_edges).T.unique(dim=1)


    def _summary(self):
        self.subgraphs = {
            "node_feat": self.data.x, # 原始圖node feature(原圖維度)
            "node_feat_imp": self.features_importance, # k個子圖分別的node feature 重要度(原圖維度)
            "edge_imp": self.edges_importance, # k個子圖分別的edge 重要度(原圖維度)
            "subgraphs_nodes": self.subgraphs_nodes, # k個子圖的node
            "subgraphs_edges": self.subgraphs_edges, # k個子圖的edge
            "subgraphs_edges_idx": self.edges_idx, # k個子圖的edge index
            "origraph_edges": self.origraph_edges # k個子圖的node在原圖中存在的所有邊
        }


    # 檢查子圖的節點，在原圖中是否存在路徑
    def _check_path(self):
        G = nx.Graph()
        G.add_nodes_from(self.subgraphs_nodes.tolist())
        G.add_edges_from(self.origraph_edges.T.tolist())

        if nx.is_connected(G):
            print(f"Success: The explainer has successfully identified a connected subgraph for the node(s) {self.node_id_explain}.")
            
        else:
            print("Reselect the subgraph!")
            self.generate_or_load_subgraphs(use_existing=False)
