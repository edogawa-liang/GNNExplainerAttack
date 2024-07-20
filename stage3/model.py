import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv, SAGEConv
import torch.nn.functional as F

# Link prediction model
class MyGNN(torch.nn.Module):
    def __init__(self, conv_type='GCN', input_dim=128, hidden_dim=128, output_dim=64):
        super(MyGNN, self).__init__()

        self.conv_type = conv_type
        self.combiner = nn.Linear(input_dim * 4, input_dim)

        # Initialize standard graph convolution layers
        if conv_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        elif conv_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, output_dim)

        elif conv_type == 'GraphSAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")

    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def get_prediction(self, node_embedding, edges):
        embedding_first_node = node_embedding[edges[0]]
        embedding_second_node = node_embedding[edges[1]]
        inner_product = torch.sum(embedding_first_node * embedding_second_node, dim=-1)
        return inner_product
