# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange

from models import inner_embedding_network
from .utils import pad_batch, unpad_batch
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
from models.digae_layers import DirectedGCNConvEncoder
import torch.nn.functional as F

def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


class DigaeSE(torch.nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, out_dim, encoder=None, decoder=None):
        super(DigaeSE, self).__init__()
        self.encoder = DirectedGCNConvEncoder(embedding_dim, hidden_dim, out_dim, alpha=0.2, beta=0.8,
                                            self_loops=True,
                                            adaptive=False) if encoder is None else encoder

    def forward(self, x, edge_index, edge_attr):
        u = x.clone()
        v = x.clone()

        s,t = self.encoder(u,v,edge_index)

        return torch.cat([s, t], dim = 1)


class Attention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    symmetric (bool):       whether K=Q in dot-product attention (default: False)
    gnn_type (str):         GNN type to use in structure extractor. (see gnn_layers.py for options)
    se (str):               type of structure extractor ("gnn", "khopgnn")
    k_hop (int):            number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
        symmetric=False, gnn_type="gcn", se="gnn", k_hop=1, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.se = se

        self.gnn_type = gnn_type
        if self.se == "khopgnn":
            self.khop_structure_extractor = KHopStructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        elif self.se == "digae":
            self.structure_extractor = DigaeSE(embed_dim, 64, embed_dim // 2)
        elif self.se == "formula-net":
            self.structure_extractor = inner_embedding_network.message_passing_gnn_sat(embed_dim, k_hop)
        else:
            self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

        # print (f"Attn network {self}")

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            edge_index,
            complete_edge_index,
            subgraph_node_index=None,
            subgraph_edge_index=None,
            subgraph_indicator_index=None,
            subgraph_edge_attr=None,
            edge_attr=None,
            ptr=None,
            return_attn=False):
        """
        Compute attention layer. 

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        complete_edge_index:        edge index from fully connected graph
        subgraph_node_index:        documents the node index in the k-hop subgraphs
        subgraph_edge_index:        edge index of the extracted subgraphs 
        subgraph_indicator_index:   indices to indicate to which subgraph corresponds to which node
        subgraph_edge_attr:         edge attributes of the extracted k-hop subgraphs
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix


        assert ptr is not None

        if complete_edge_index is None:
            # print (f"ptr: {ptr}")
            complete_edge_index = ptr_to_complete_edge_index(ptr.cpu()).cuda()


        v = self.to_v(x)

        # Compute structure-aware node embeddings 
        if self.se == 'khopgnn': # k-subgraph SAT
            x_struct = self.khop_structure_extractor(
                x=x,
                edge_index=edge_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_attr=subgraph_edge_attr,
            )
        else: # k-subtree SAT
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
            # can set x_struct = x here for normal transformer
            # x_struct = x


        # print (f"x.shape {x.shape} x_struct {x_struct.shape}")
        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            # todo AMR, Q is using nodes (x_struct), K is using relation R(i,j)
            # todo return tuple (K, Q) as needed? K are r(i,j)
            # todo keep Q_out, Q_in for target/source embeddings, only need one K
            qk = self.to_qk(x_struct).chunk(2, dim=-1)
        
        # Compute complete self-attention
        attn = None

        # print (f"qk {qk[0].shape, qk[1].shape}, v: {v.shape}")
        # print (f"qk {qk}, v: {v}")

        #passed in as None for ppa and code?
        # MUCH faster than if no complete edge index, pad batch from self_attn is extremely slow


        #for DAG, complete edge index should be
        if complete_edge_index is not None:
            # print ("complete")

            #todo AMR implementation:

            #todo q with h_i, k,v with r(i,j)
            #todo message is from -> to (j -> i)

            #todo edge index will be "from" all edge relations "to" nodes. Each edge relation must map to it's source/target so will have 2 * num_edges edges

            #todo make separate "complete_edge_index" which has "from" relations "to" source nodes, and "from" relations to the corresponding "target" nodes
            #todo can then use these edge indices to do two separate propagate calls, then combine the outputs g_i_out, g_i_in afterwards



            out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)

            # checking output below, confirm q/k is in right order with qk_j = qk[0][j], qk_i = qk[1][i]

            # out = self.propagate(complete_edge_index, v=v, qk=(qk[1],qk[0]), edge_attr=None, size=None,
            #                      return_attn=return_attn)

            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    complete_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)

            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """


        #todo AMR make sure size_i isn't breaking softmax for non-complete index

        # size_i = max(index) + 1 # from torch_geometric docs? todo test correct

        # qk_j is keys i.e. message "from" j, qk_i maps to queries i.e. messages "to" i

        # index maps to the "to"/ i values i.e. index[i] = 3 means i = 3, and len(index) is the number of messages
        # i.e. index will be 0,n repeating n times (if complete_edge_index is every combination of nodes)

        # print (f"qkj: {qk_j}, qki: {qk_i}, vj: {v_j}")

        # print (f"message: v_j {v_j.shape}, qk_j: {qk_j.shape}, index: {index}, ptr: {ptr}, size_i: {size_i}")

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)

        # print (f"message after: v_j {v_j.shape}, qk_j: {qk_j.shape}, index: {index}, ptr: {ptr}, size_i: {size_i}")

        # sum over dimension, giving n h shape
        attn = (qk_i * qk_j).sum(-1) * self.scale

        # print (attn.shape)

        if edge_attr is not None:
            attn = attn + edge_attr

        # index gives what to softmax over

        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, return_attn=False):
        """ Self attention which can return the attn """ 

        # print ([q.shape for q in qk], v.shape)

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None


class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=1,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x


class KHopStructureExtractor(nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree (True)
    """
    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3, batch_norm=True,
            concat=True, khopgnn=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn

        self.batch_norm = batch_norm

        self.structure_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            concat=False,
            khopgnn=True,
            **kwargs
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * embed_dim)

        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
            subgraph_indicator_index=None, subgraph_node_index=None,
            subgraph_edge_attr=None):

        x_struct = self.structure_extractor(
            x=x[subgraph_node_index],
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            agg="sum",
        )
        x_struct = torch.cat([x, x_struct], dim=-1)
        if self.batch_norm:
            x_struct = self.bn(x_struct)
        x_struct = self.out_proj(x_struct)

        return x_struct


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Structure-Aware Transformer layer, made up of structure-aware self-attention and feed-forward network.

    Args:
    ----------
        d_model (int):      the number of expected features in the input (required).
        nhead (int):        the number of heads in the multiheadattention models (default=8).
        dim_feedforward (int): the dimension of the feedforward network model (default=512).
        dropout:            the dropout value (default=0.1).
        activation:         the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default: relu).
        batch_norm:         use batch normalization instead of layer normalization (default: True).
        pre_norm:           pre-normalization or post-normalization (default=False).
        gnn_type:           base GNN model to extract subgraph representations.
                            One can implememnt customized GNN in gnn_layers.py (default: gcn).
        se:                 structure extractor to use, either gnn or khopgnn (default: gnn).
        k_hop:              the number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", se="gnn", k_hop=2, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
            bias=False, gnn_type=gnn_type, se=se, k_hop=k_hop, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None,
            subgraph_indicator_index=None,
            edge_attr=None, degree=None, ptr=None,
            return_attn=False,
        ):

        # print (f"running")
        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        if degree is not None:
            x2 = degree.unsqueeze(-1) * x2
        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x
