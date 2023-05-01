# -*- coding: utf-8 -*-
from torch_geometric.data import Data
from einops import rearrange
import copy
import math
import torch
from torch import nn
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer, AMREncoderLayer, MPAttention, MPAttentionAggr
from einops import repeat


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index

class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):

        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, 
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,k_hop=2,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)
        # print ("r_inductunning")
        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Embedding(abs_pe_dim, d_model)
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, k_hop=k_hop,**kwargs)

        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None

        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
            
        if self.abs_pe and abs_pe is not None:
            # abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            # print (data.ptr.shape)
            # print (data.ptr)
            # print (len(data.batch))


            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output, 
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        # print (f"Output shape: {output.shape} \n")
        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                # print (output.shape)
                output = output[-bsz:]
                # print (output.shape)
            else:
                output = gnn.global_max_pool(output, data.batch)

                # output_1 = self.pooling(output, data.batch)
                # output_2 = gnn.global_max_pool(output, data.batch)
                # output = torch.cat([output_1, output_2], dim=1)

        return output

        # if self.max_seq_len is not None:
        #     pred_list = []
        #     for i in range(self.max_seq_len):
        #         pred_list.append(self.classifier[i](output))
        #     return pred_list
        #
        # return self.classifier(output)



class AMREncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, edge_index_source, edge_index_target, softmax_idx,
                 edge_attr=None,
                ptr=None, return_attn=False):


        xs, xt = x,x

        for mod in self.layers:
            xs, xt = mod(xs, xt, edge_index, edge_index_source, edge_index_target,
                         softmax_idx=softmax_idx,
                         edge_attr=edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        # if self.norm is not None:
        #     output = self.norm(output)



        return torch.cat([xs,xt], dim = 1)


class AMRTransformer(nn.Module):
    def __init__(self, in_size,  d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 layer_norm=False, abs_pe=False, abs_pe_dim=0,
                 use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True,
                 global_pool='mean', device='cuda',**kwargs):
        super().__init__()

        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)
        # print ("r_inductunning")
        self.abs_pe = abs_pe

        self.abs_pe_dim = abs_pe_dim

        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Embedding(abs_pe_dim, d_model)
            # self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        self.device = device

        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = 0

        encoder_layer = AMREncoderLayer(
            d_model, num_heads, dim_feedforward,
            dropout, layer_norm=layer_norm,
            edge_dim=edge_dim,**kwargs)

        self.encoder = AMREncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model * 2))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.attn_pool = MPAttentionAggr(embed_dim=d_model * 2, num_heads=num_heads, dropout=dropout)

    def forward(self, data, return_attn=False):

        x, edge_index, edge_attr, softmax_idx = data.x, data.edge_index, data.edge_attr, data.softmax_idx



        # from_ids = [edge_index[0][i] for i in torch.arange(edge_index.shape[1])]
        # to_ids =  [edge_index[1][i] for i in torch.arange(edge_index.shape[1])]

        # from_ids = torch.index_select(edge_index[0], 0, torch.arange(edge_index.shape[1]))
        # to_ids = torch.index_select(edge_index[1], 0, torch.arange(edge_index.shape[1]))
        range_ids = torch.arange(edge_index.shape[1]).to(self.device)

        edge_index_source = torch.stack([range_ids,edge_index[0]], dim = 0)
        edge_index_target = torch.stack([range_ids,edge_index[1]], dim = 0)


        # softmax_idx = []
        # cur_idx = 0
        #
        # for ind in edge_index[0]:
        #     if ind < data.ptr[cur_idx+1]:
        #         softmax_idx.append(cur_idx)
        #     else:
        #         cur_idx += 1
        #         softmax_idx.append(cur_idx)


        # edge_index_source = torch.LongTensor([[i for i in torch.arange(edge_index.shape[1])], [edge_index[0][i] for i in torch.arange(edge_index.shape[1])]]).to(self.device)
        # edge_index_target = torch.LongTensor([[i for i in torch.arange(edge_index.shape[1])], [edge_index[1][i] for i in torch.arange(edge_index.shape[1])]]).to(self.device)

        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None


        output = self.embedding(x)


        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None


        output = self.encoder(
            x=output,
            edge_index=edge_index,
            edge_index_source=edge_index_source,
            edge_index_target=edge_index_target,
            softmax_idx=softmax_idx,
            edge_attr=edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )


        # if self.global_pool == 'cls' and self.use_global_pool:
        #     bsz = len(data.ptr) - 1
        #
        #     # complete_edge_index = ptr_to_complete_edge_index(data.ptr.cpu()).cuda()
        #
        #     new_index = torch.vstack((torch.arange(data.num_nodes).to(x), data.batch + data.num_nodes))
        #     new_index2 = torch.vstack((new_index[1], new_index[0]))
        #     idx_tmp = torch.arange(data.num_nodes, data.num_nodes + len(data.ptr) - 1).to(data.batch)
        #     new_index3 = torch.vstack((idx_tmp, idx_tmp))
        #
        #     # complete_edge_index = torch.cat((
        #     #     complete_edge_index, new_index, new_index2, new_index3), dim=-1)
        #
        #     complete_edge_index = torch.cat((
        #          new_index, new_index2, new_index3), dim=-1)
        #
        #     cls_tokens = repeat(self.cls_token, '() d -> b d', b=len(data.ptr) - 1)
        #
        #     output = torch.cat((output, cls_tokens))


        if self.use_global_pool:
            if self.global_pool == 'cls':
                bsz = len(data.ptr) - 1
                output, attn = self.attn_pool(data=Data(x=output, edge_index=data.edge_index, ptr=data.ptr, batch=data.batch, num_nodes = data.num_nodes), return_attn=True)
                output = output[-bsz:]
            else:
                output = gnn.global_max_pool(output, data.batch)

        return output




