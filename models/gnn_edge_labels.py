import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GINEConv, GINConv




from torch.nn.functional import dropout
import torch.nn.functional as F
import numpy as np


# F_o summed over children
class Child_Aggregation_edges(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum', flow='target_to_source')

        self.mlp = Seq(Dropout(), Linear(2 * in_channels + 32, out_channels),
                       ReLU(), Dropout(),
                       Linear(out_channels, out_channels))

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


# F_i summed over parents
class Parent_Aggregation_edges(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum', flow='source_to_target')
        self.mlp = Seq(Dropout(), Linear(2 * in_channels + 32, out_channels),
                       ReLU(), Dropout(),
                       Linear(out_channels, out_channels))

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


class Final_Agg_edges(nn.Module):
    def __init__(self, embedding_dim):
        super(Final_Agg_edges, self).__init__()
        # self.fc = nn.Linear(embedding_dim * 3, embedding_dim * 2)
        # self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        x = torch.relu(self.fc(dropout(x)))
        return x


class F_x_module_(nn.Module):
    def __init__(self, input_shape, embedding_dim):
        super(F_x_module_, self).__init__()
        self.fc1 = nn.Linear(input_shape, embedding_dim)
    def forward(self, x):
        return self.fc1(x)


class F_c_module_(nn.Module):
    def __init__(self, input_shape):
        super(F_c_module_, self).__init__()

        self.fc1 = nn.Linear(input_shape, input_shape // 2)

        self.fc2 = nn.Linear(input_shape // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(dropout(x)))

        return torch.sigmoid(self.fc2(dropout(x)))


class message_passing_gnn_edges(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations):
        super(message_passing_gnn_edges, self).__init__()
        self.num_iterations = num_iterations
        # self.initial_encoder = F_x_module_(input_shape, embedding_dim)
        self.initial_encoder = torch.nn.Embedding(input_shape, embedding_dim)
        #assume max 200 children
        self.edge_encoder = nn.Embedding(200, 32)
        self.parent_agg = Parent_Aggregation_edges(embedding_dim, embedding_dim)
        self.child_agg = Child_Aggregation_edges(embedding_dim, embedding_dim)
        self.final_agg = Final_Agg_edges(embedding_dim)

    def forward(self, batch):# nodes, edges, edge_attr, batch=None):
        nodes = batch.x
        edges = batch.edge_index
        edge_attr = batch.edge_attr
        nodes = self.initial_encoder(nodes)
        edge_encodings = self.edge_encoder(edge_attr)#torch.transpose(edge_attr, 0, 1))

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges, edge_encodings)
            fo_sum = self.child_agg(nodes, edges, edge_encodings)
            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update

        return gmp(nodes, batch.batch)


class message_passing_gnn_induct(nn.Module):

    def __init__(self, input_shape, embedding_dim, num_iterations, device):
        super(message_passing_gnn_induct, self).__init__()

        self.device = device

        self.num_iterations = num_iterations

        self.initial_encoder = F_x_module_(input_shape, embedding_dim).to(device)

        self.parent_agg = Parent_Aggregation(embedding_dim, embedding_dim).to(device)

        self.child_agg = Child_Aggregation(embedding_dim, embedding_dim).to(device)

        self.final_agg = Final_Agg(embedding_dim).to(device)

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim * 2, 1, stride=1).to(device)

    def forward(self, nodes, edges, edge_attr,  batch=None):
        nodes = self.initial_encoder(nodes)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)

            fo_sum = self.child_agg(nodes, edges)

            node_update = self.final_agg(torch.cat([nodes, fi_sum, fo_sum], axis=1).to(self.device))

            nodes = nodes + node_update

        nodes = nodes.unsqueeze(-1)

        nodes = self.conv1(nodes)

        nodes = nodes.squeeze(-1)

#        g_embedding = torch.cat([gmp(nodes, batch), gap(nodes, batch)], dim=1)  # gmp(nodes, batch)
        #return embeddings for each node which is a variable
        return nodes


class message_passing_gnn_edges_gine(nn.Module):

    def __init__(self, input_shape, embedding_dim, num_iterations, device):
        super(message_passing_gnn_edges_gine, self).__init__()

        self.device = device

        self.num_iterations = num_iterations

        self.initial_encoder = F_x_module_(input_shape, embedding_dim).to(device)

        self.edge_encoder = F_x_module_(1, embedding_dim).to(device)

        #self.parent_agg = Parent_Aggregation_edges(embedding_dim, embedding_dim).to(device)

        self.mlp_1 = Seq(Dropout(), Linear(1 * embedding_dim, embedding_dim),
                       ReLU(), Dropout(),
                       Linear(embedding_dim, embedding_dim))

        self.mlp_2 = Seq(Dropout(), Linear(1 * embedding_dim, embedding_dim),
                       ReLU(), Dropout(),
                       Linear(embedding_dim, embedding_dim))


        self.parent_agg = GINEConv(self.mlp_1, eps=0., train_eps=False, flow='source_to_target').to(device)
        self.child_agg = GINEConv(self.mlp_2, eps=0., train_eps=False, flow='target_to_source').to(device)
        # self.child_agg = Child_Aggregation_edges(embedding_dim, embedding_dim).to(device)

        self.final_agg = Final_Agg_edges(embedding_dim).to(device)

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim * 2, 1, stride=1).to(device)

    def forward(self, nodes, edges, edge_attr, batch=None):
        nodes = self.initial_encoder(nodes)
        edge_encodings = self.edge_encoder(torch.unsqueeze(edge_attr, 1).float())#torch.transpose(edge_attr, 0, 1))

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges, edge_encodings)

            fo_sum = self.child_agg(nodes, edges, edge_encodings)

            node_update = self.final_agg(torch.cat([nodes, fi_sum, fo_sum], axis=1).to(self.device))

            nodes = nodes + node_update

        nodes = nodes.unsqueeze(-1)

        nodes = self.conv1(nodes)

        nodes = nodes.squeeze(-1)

        g_embedding = torch.cat([gmp(nodes, batch), gap(nodes, batch)], dim=1)  # gmp(nodes, batch)
        return g_embedding
