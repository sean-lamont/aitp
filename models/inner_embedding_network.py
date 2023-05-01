import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch.nn.functional import dropout
import torch.nn.functional as F
import numpy as np


# F_o summed over children
class Child_Aggregation(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum', flow='target_to_source')

        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        # self.mlp = Seq(Linear(2 * in_channels, out_channels),
        #                nn.BatchNorm1d(out_channels),
        #                ReLU(),
        #                Linear(out_channels, out_channels),
        #                nn.BatchNorm1d(out_channels))
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def forward(self, x, edge_index):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)

        deg_inv = 1. / deg

        deg_inv[deg_inv == float('inf')] = 0

        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)


# F_i summed over parents
class Parent_Aggregation(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum', flow='source_to_target')

        self.mlp = Seq(Dropout(), Linear(2 * in_channels, out_channels),
                       ReLU(), Dropout(),
                       Linear(out_channels, out_channels))

        # self.mlp = Seq(Linear(2 * in_channels, out_channels),
        #                nn.BatchNorm1d(out_channels),
        #                ReLU(),
        #                Linear(out_channels, out_channels),
        #                nn.BatchNorm1d(out_channels))


    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)

        return self.mlp(tmp)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # edge index 1 for degree wrt parents
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype)

        deg_inv = 1. / deg

        deg_inv[deg_inv == float('inf')] = 0

        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)

class Final_Agg_induct(nn.Module):
    def __init__(self, embedding_dim):
        super(Final_Agg_induct, self).__init__()

        self.fc = nn.Linear(embedding_dim * 3, embedding_dim * 2)
        #
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)

        # self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc2(dropout(torch.relu(self.fc(dropout(x)))))
        # x = torch.relu(self.fc(dropout(x)))
        return x

class Final_Agg(nn.Module):
    def __init__(self, embedding_dim):
        super(Final_Agg, self).__init__()

        # self.fc = nn.Linear(embedding_dim * 3, embedding_dim * 2)
        #
        # self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)

        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x = self.fc2(dropout(torch.relu(self.fc(dropout(x)))))
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

        self.fc1 = nn.Linear(input_shape, input_shape)

        self.fc2 = nn.Linear(input_shape, 1)

        self.bn = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))

        return torch.sigmoid(self.fc2(x))

class message_passing_gnn_sat(nn.Module):

    def __init__(self, embedding_dim, num_iterations):
        super(message_passing_gnn_sat, self).__init__()

        self.num_iterations = num_iterations

        self.parent_agg = Parent_Aggregation(embedding_dim, embedding_dim)

        self.child_agg = Child_Aggregation(embedding_dim, embedding_dim)

        self.final_agg = Final_Agg(embedding_dim)

    def forward(self, nodes, edges, batch=None):

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)

            fo_sum = self.child_agg(nodes, edges)

            node_update = self.final_agg(nodes + fi_sum + fo_sum)

            nodes = nodes + node_update

        return nodes
        # return gmp(nodes,batch)
#






class FormulaNet(nn.Module):

    def __init__(self, input_shape, embedding_dim, num_iterations):
        super(FormulaNet, self).__init__()

        self.num_iterations = num_iterations

        self.initial_encoder = F_x_module_(input_shape, embedding_dim)

        self.parent_agg = Parent_Aggregation(embedding_dim, embedding_dim)

        self.child_agg = Child_Aggregation(embedding_dim, embedding_dim)

        self.final_agg = Final_Agg(embedding_dim)

    # def forward(self, nodes, edges, batch=None):
    def forward(self, data):
        nodes = data.x
        edges = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        nodes = self.initial_encoder(nodes)


        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)

            fo_sum = self.child_agg(nodes, edges)

            node_update = self.final_agg(nodes + fi_sum + fo_sum)

            nodes = nodes + node_update

        return gmp(nodes,batch)




class message_passing_gnn_(nn.Module):

    def __init__(self, input_shape, embedding_dim, num_iterations, device):
        super(message_passing_gnn_, self).__init__()

        self.device = device

        self.num_iterations = num_iterations

        self.initial_encoder = F_x_module_(input_shape, embedding_dim).to(device)

        self.parent_agg = Parent_Aggregation(embedding_dim, embedding_dim).to(device)

        self.child_agg = Child_Aggregation(embedding_dim, embedding_dim).to(device)

        self.final_agg = Final_Agg(embedding_dim).to(device)

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim * 2, 1, stride=1).to(device)

    # def forward(self, nodes, edges, batch=None):
    def forward(self, data):
        nodes = data.x
        edges = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        nodes = self.initial_encoder(nodes)


        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)

            fo_sum = self.child_agg(nodes, edges)

            # node_update = self.final_agg(torch.cat([nodes, fi_sum, fo_sum], axis=1).to(self.device))

            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update

        # nodes = nodes.unsqueeze(-1)
        #
        # nodes = self.conv1(nodes)
        #
        # nodes = nodes.squeeze(-1)

        # g_embedding = torch.cat([gmp(nodes, batch), gap(nodes, batch)], dim=1)  # gmp(nodes, batch)
        # return g_embedding
        return gmp(nodes,batch)
#define GNN for induction (term) network

class message_passing_gnn_induct(nn.Module):

    def __init__(self, input_shape, embedding_dim, num_iterations, device):
        super(message_passing_gnn_induct, self).__init__()

        self.device = device

        self.num_iterations = num_iterations

        self.initial_encoder = F_x_module_(input_shape, embedding_dim).to(device)

        self.parent_agg = Parent_Aggregation(embedding_dim, embedding_dim).to(device)

        self.child_agg = Child_Aggregation(embedding_dim, embedding_dim).to(device)

        self.final_agg = Final_Agg_induct(embedding_dim).to(device)

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim * 2, 1, stride=1).to(device)

    def forward(self, nodes, edges, batch=None):
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



# additional networks for meta graph learning


# define H_s as per inductive link prediction paper
class StructureEmbeddingSource(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(StructureEmbeddingSource, self).__init__()
        self.fc = nn.Linear(num_nodes, embedding_dim)

    def forward(self, x):
        return self.fc(x)


class StructureEmbeddingTarget(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(StructureEmbeddingTarget, self).__init__()
        self.fc = nn.Linear(num_nodes, embedding_dim)

    def forward(self, x):
        return self.fc(x)


def sp_to_torch(sparse):
    coo = sparse.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))  # .to_dense()


# regulariser
def phi(x, gamma, b):
    return (1. / gamma) * torch.log(1. + torch.exp(-1. * gamma * x + b))


# #used to scale negative samples
# def alpha(beta, distance):
#     return

def loss_similarity(sim, y, phi_1, phi_2, alpha=None):
    return y * phi_1(sim) + (1. - y) * (phi_2(-1. * sim))
