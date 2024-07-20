import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class FBGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(128, 64, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(64, 32, cached=True,
                             normalize=True)
        self.conv3 = GCNConv(32, 4, cached=True,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x
    


class GitGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(128, 64, cached=True,
                             normalize=True)
        # self.conv2 = GCNConv(64, 32, cached=True,
        #                      normalize=True)
        self.conv3 = GCNConv(64, 2, cached=True,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        # x = F.dropout(x, p=0.4, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x