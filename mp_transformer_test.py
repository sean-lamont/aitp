import torch
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as gloader

import models.graph_transformers.SAT.sat.models
import models.graph_transformers.SAT.sat.layers

from torch.utils.data import DataLoader, TensorDataset
# from torch_geometric.data import Data
import pickle
import json
from data.hol4.ast_def import *


#%%



#%%
with open("data/hol4/data/torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)
#%%
with open("data/hol4/data/train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)
#%%
with open("data/hol4/data/adjusted_db.json") as f:
    db = json.load(f)
#%%
tokens = list(
    set([token.value for polished_goal in db.keys() for token in polished_to_tokens_2(polished_goal)]))
#%%

# from torchtext.vocab import build_vocab_from_iterator
# def build_vocab(l):
#     for token in l:
#         yield [token]
#
# vocab = build_vocab_from_iterator(build_vocab(tokens), specials=["<UNK>"], min_freq=0)
# vocab.set_default_index(vocab["<UNK>"])


#%%
# train_seq = []
#
# max_len = 1024
#
#
# for i, (goal, premise, y) in enumerate(train):
#     train_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))
#
# val_seq = []
# for i, (goal, premise, y) in enumerate(val):
#     val_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))
#
# test_seq = []
# for i, (goal, premise, y) in enumerate(test):
#     test_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))
#
#%%

# train_goals = []
# train_premises = []
# train_targets = []
#
# for goal, premise, y in train_seq:
#     train_goals.append(goal)
#     train_premises.append(premise)
#     train_targets.append(y)
#
#
# val_goals = []
# val_premises = []
# val_targets = []
#
# for goal, premise, y in val_seq:
#     val_goals.append(goal)
#     val_premises.append(premise)
#     val_targets.append(y)
#
# test_goals = []
# test_premises = []
# test_targets = []
#
# for goal, premise, y in test_seq:
#     test_goals.append(goal)
#     test_premises.append(premise)
#     test_targets.append(y)

#%%
def vectorise(goal_list, premise_list, target_list, max_len=1024):
    idx_list = [vocab(toks) for toks in goal_list]
    X_G = [sample+([0]* (max_len-len(sample))) if len(sample)<max_len else sample[:max_len] for sample in idx_list]
    idx_list = [vocab(toks) for toks in premise_list]
    X_P = [sample+([0]* (max_len-len(sample))) if len(sample)<max_len else sample[:max_len] for sample in idx_list]
    return torch.tensor(X_G, dtype=torch.int32), torch.tensor(X_P, dtype=torch.int32), torch.tensor(target_list, dtype=torch.long)

#%%
# train_dataset = vectorise(train_goals, train_premises, train_targets)
# val_data = vectorise(val_goals, val_premises, val_targets)

#%%
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEmbedding(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)

        # self.initial_encoder = inner_embedding_network.F_x_module_(ntoken, d_model)

        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_embedding(model, input, max_seq_len=129):
    out = model(input)#, src_mask)
    out = torch.transpose(out,1,2)
    gmp = nn.MaxPool1d(max_seq_len, stride=1)
    return gmp(out).squeeze(-1)#orch.cat([gmp(out).squeeze(-1), torch.sum(out,dim=2)], dim = 1)

#%%
from models import inner_embedding_network
def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


#run_edges(1e-3, 0, 20, 1024, 64, 0, False)
#run_2(1e-3, 0, 20, 1024, 64, 4, False)

def accuracy_transformer(model_1, model_2,batch, fc):
    g,p,y = batch
    batch_size = len(g)

    embedding_1 = gen_embedding(model_1, g.to(device))
    embedding_2 = gen_embedding(model_2, p.to(device))

    preds = fc(torch.cat([embedding_1, embedding_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(y).to(device)) / len(y)

def run_transformer_pretrain(step_size, decay_rate, num_epochs, batch_size, embedding_dim, save=False):

    # loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    # val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    G,P,Y = train_dataset

    dataset = TensorDataset(G,P,Y)
    # batch_size = 50
    loader = DataLoader(dataset, batch_size=batch_size)

    V_G, V_P, V_Y = val_data
    val_dataset = TensorDataset(V_G, V_P, V_Y)

    val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_1 = TransformerEmbedding(ntoken=len(vocab), d_model=embedding_dim, nhead=1, d_hid=embedding_dim, nlayers=1).to(device)
    model_2 = TransformerEmbedding(ntoken=len(vocab), d_model=embedding_dim, nhead=1, d_hid=embedding_dim, nlayers=1).to(device)

    fc = inner_embedding_network.F_c_module_(embedding_dim * 2).to(device)

    op_1 =torch.optim.Adam(model_1.parameters(), lr=step_size)
    op_2 =torch.optim.Adam(model_2.parameters(), lr=step_size)
    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    val_losses = []
    best_acc = 0.


    for j in range(num_epochs):
        print (f"Epoch: {j}")

        for batch_idx, (g,p,y) in enumerate(loader):
            # op_enc.zero_grad()
            op_1.zero_grad()
            op_2.zero_grad()
            op_fc.zero_grad()

            embedding_1 = gen_embedding(model_1, g.to(device))
            embedding_2 = gen_embedding(model_2, p.to(device))

            preds = fc(torch.cat([embedding_1, embedding_2], axis=1))

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), torch.LongTensor(y).to(device))

            loss.backward()

            op_1.step()
            op_2.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)

            if batch_idx % 100 == 0:

                validation_loss = accuracy_transformer(model_1, model_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                val_losses.append((validation_loss.detach(), j, batch_idx))

                val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(validation_loss.detach()))

    return training_losses, val_losses

# run_transformer_pretrain(1e-3, 0, 40, 32, 128, 2)#, save=True)


#%%
# G,P,Y = train_dataset


#%%
# train_goals[0]
#%%
'''
Vectorise sequence without requiring a maximum length
'''
def to_mp_data(data_list, vocab):
    return [Data(torch.tensor(vocab(toks), dtype=torch.int32)) for toks in data_list]



#%%
# len(train_goals[])
#%%
# train_graphs = to_mp_data(train_goals, vocab)
#
#%%
import torch_geometric.nn as gnn
from einops import rearrange, repeat
import torch_geometric.utils as utils

'''
Implementation of standard transformer through message passing. Generates a fully connected graph on input sequence
and performs self attention using message passing.

Batching is done through PyG, with a batch consisting only of (batch_size, d_model),
as opposed to standard (batch_size, max_seq_len, d_model)
'''

def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


class MPAttention(gnn.MessagePassing):


    def __init__(self, embed_dim,  num_heads=8, dropout=0., bias=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')

        self.embed_dim = embed_dim
        self.bias = bias

        head_dim = embed_dim // num_heads

        # print (embed_dim, num_heads, head_dim)


        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads

        self.scale = head_dim ** -0.5


        self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)

        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # self.ffn = torch.nn.Sequential(nn.Linear(embed_dim, embed_dim * 4, bias=bias),
        #                                nn.ReLU(),
        #                                nn.Linear(embed_dim * 4, embed_dim * 2))

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

        self.attn_sum = None

        # print (f"Attn network {self}")

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_qk.weight)

        if self.bias:
            nn.init.xavier_uniform_(self.to_qk.weight)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x,
                edge_index,
                complete_edge_index,
                edge_attr=None,
                ptr=None,
                return_attn=False):

        assert ptr is not None

        # if edge_index is None:
        #     edge_index = ptr_to_complete_edge_index(ptr.cpu()).cuda()



        qk = self.to_qk(x).chunk(2, dim=-1)

        v = self.to_v(x)

        attn = None


        out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                             return_attn=return_attn)

        # print (out.shape)

        out = rearrange(out, 'n h d -> n (h d)')


        # if return_attn:
        #     attn = self._attn
        #     self._attn = None
        #     attn = torch.sparse_coo_tensor(
        #         complete_edge_index,
        #         attn,
        #     ).to_dense().transpose(0, 1)


        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """



        print (f"v_j {v_j.shape}, qk_j: {qk_j.shape}, qk_i: {qk_i.shape}")

        print (f"index {index.shape}\n\n")

        #todo AMR make sure size_i isn't breaking softmax for non-complete index

        # size_i = max(index) + 1 # from torch_geometric docs? todo test correct

        # qk_j is keys i.e. message "from" j, qk_i maps to queries i.e. messages "to" i

        # index maps to the "to"/ i values i.e. index[i] = 3 means i = 3, and len(index) is the number of messages
        # i.e. index will be 0,n repeating n times (if complete_edge_index is every combination of nodes)


        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)

        print (f"post arrange v_j {v_j.shape}, qk_j: {qk_j.shape}, qk_i: {qk_i.shape}")
        # sum over dimension, giving n h shape
        attn = (qk_i * qk_j).sum(-1) * self.scale

        print (f"attn shape {attn.shape}")

        if edge_attr is not None:
            attn = attn + edge_attr

        # index gives what to softmax over

        attn = utils.softmax(attn, index, ptr, size_i)

        print (f"attn shape after softmax {attn.shape}")

        if return_attn:
            self._attn = attn

        attn = self.attn_dropout(attn)

        msg = v_j * attn.unsqueeze(-1)

        print (f"msg shape {msg.shape}")

        return msg

class MPTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 **kwargs):


        # print (nhead)
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = MPAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                     bias=False, **kwargs)

        self.batch_norm = batch_norm
        self.pre_norm = pre_norm

        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index,complete_edge_index,
                ptr=None,
                return_attn=False,
                ):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            complete_edge_index,
            ptr=ptr,
            return_attn=return_attn
        )

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


class MPPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, ptr):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        pe_ptr = torch.cat([self.pe[:(ptr[i+1] - ptr[i])] for i in range(len(ptr) - 1)], dim = 0)

        # print (pe_ptr.shape)

        return pe_ptr

class MPTransformerEncoder(nn.TransformerEncoder):

    def forward(self, x, edge_index, complete_edge_index, edge_attr=None,ptr=None, return_attn=False):

        output = x

        for mod in self.layers:

            output = mod(output,
                         edge_index=edge_index,
                         complete_edge_index=complete_edge_index,
                         ptr=ptr,
                         return_attn=return_attn
                         )

        if self.norm is not None:
            output = self.norm(output)

        return output


class MPTransformer(nn.Module):

    def __init__(self, in_size, d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 batch_norm=False, pe=False,
                 in_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):

        super().__init__()

        # print (f"insize {in_size}, d_model {d_model}, num_heads: {num_heads}")
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)

        # if pos_encoder:
        #     self.pos_encoder = MPPositionalEncoding(d_model, dropout)
        # else:
        #     self.pos_encoder = None

        self.pe = pe

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)


        encoder_layer = MPTransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_norm=batch_norm,**kwargs)

        self.encoder = MPTransformerEncoder(encoder_layer, num_layers)

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

    def forward(self, data, return_attn=False):

        output = data.x

        ptr = data.ptr

        complete_edge_index = data.complete_edge_index

        # if hasattr(data, 'edge_index'):
        #     edge_index = data.edge_index
        # else:
        #     edge_index = ptr_to_complete_edge_index(ptr.cpu()).cuda()


        output = self.embedding(output)


        if self.pe:
            # print (data.pe[0], output[0], data.pe.shape, output.shape)
            output = output + data.pe


        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1

            # if edge_index is not None:
            #     new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
            #     new_index2 = torch.vstack((new_index[1], new_index[0]))
            #     idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
            #     new_index3 = torch.vstack((idx_tmp, idx_tmp))
            #     edge_index = torch.cat((
            #         edge_index, new_index, new_index2, new_index3), dim=-1)

            degree = None

            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)

            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            edge_index = None,
            complete_edge_index=complete_edge_index,
            ptr=data.ptr,
            return_attn=return_attn
        )

        if self.use_global_pool:

            if self.global_pool == 'cls':
                output = output[-bsz:]

            else:
                # output_1 = self.pooling(output, data.batch)
                output = gnn.global_max_pool(output, data.batch)
                # output = torch.cat([output_1, output_2], dim=1)

        return output




'''
Transformer encoder for nested tensor input, requires no padding
'''

class NestedTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", layer_norm=True, pre_norm=False,
                 **kwargs):

        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,dropout=dropout, batch_first=True)

        self.layer_norm = layer_norm
        self.pre_norm = pre_norm

        if layer_norm:
            self.norm1 = nn.LayerNorm(d_model)#(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(x,x,x)

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


class NestedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, ptr):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        pe_ptr = torch.cat([self.pe[:(ptr[i+1] - ptr[i])] for i in range(len(ptr) - 1)], dim = 0)

        # print (pe_ptr.shape)

        return pe_ptr

class NestedTransformerEncoder(nn.TransformerEncoder):

    def forward(self, x, return_attn=False):

        output = x

        for mod in self.layers:

            output = mod(output, return_attn=return_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output


class NestedTransformer(nn.Module):

    def __init__(self, in_size, d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 layer_norm=False, pe=False,
                 in_embed=True, use_global_pool=True,
                 global_pool='mean', **kwargs):

        super().__init__()

        self.pe = pe

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)


        encoder_layer = NestedTransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm=layer_norm,**kwargs)

        self.encoder = NestedTransformerEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool

        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool

        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool

        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None

        self.use_global_pool = use_global_pool

    def forward(self, x, return_attn=False):

        output = self.embedding(x)


        # if self.pe:
            # print (data.pe[0], output[0], data.pe.shape, output.shape)
            # output = output +


        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = x.shape(0)

            # if edge_index is not None:
            #     new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
            #     new_index2 = torch.vstack((new_index[1], new_index[0]))
            #     idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
            #     new_index3 = torch.vstack((idx_tmp, idx_tmp))
            #     edge_index = torch.cat((
            #         edge_index, new_index, new_index2, new_index3), dim=-1)

            degree = None

            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)

            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            return_attn=return_attn
        )

        if self.use_global_pool:

            if self.global_pool == 'cls':
                output = output[-bsz:]

            else:
                # output_1 = self.pooling(output, data.batch)
                # output = gnn.global_max_pool(output, data.batch)
                # output = gnn.global_max_pool(output, data.batch)
                # output = torch.cat([output_1, output_2], dim=1)
                output = torch.sum(output, dim=1)

        return output
#%%

from time import time
from utils.viz_net_torch import make_dot
#%%
# with open("mp_transformer_test_old_new.pk", "wb") as f:
#     pickle.dump((G, train_graphs), f)
#%%
# with open("mp_transformer_test_old_new.pk", "rb") as f:
#     G, train_graphs = pickle.load(f)

#%%

#%%

# start = time()
#
# loader = iter(gloader(train_graphs, batch_size=3))
# batch = next(loader)
# batch.complete_edge_index = ptr_to_complete_edge_index(batch.ptr)
# pe = MPPositionalEncoding(128)
# batch.pe = pe(batch.x, batch.ptr).squeeze(1)
#
# print (f"Data load time {time() - start}")
#
#
# model = MPTransformer(in_size=len(vocab), d_model=128, dim_feedforward=128, num_layers=1,num_heads=1,in_embed=True,dropout=0.,max_seq_len=None,batch_norm=False,pe=True, global_pool='max').to(device)
#
# print (f"Model define time {time() - start}")
# start = time()
#
# embedding = model(batch.cuda())
#
# print (f"Model run time {time() - start}")
#
#
# start = time()
#
# loss = torch.sum(embedding)
#
# g = make_dot(loss)
# g.view()
#
# loss.backward()
#
# print (f"Model backward time: {time() - start}")

#%%
# mem 4644, 1.19 time
#%%




# start = time()
#
# old_loader = iter(DataLoader(G, batch_size=32))
# old_batch = next(old_loader)
#
# print (f"Data load time {time() - start}")
#
# start = time()
#
# tf_model = TransformerEmbedding(ntoken=len(vocab), d_model=128, nhead=1, d_hid=128, nlayers=1).to(device)
#
# print (f"Model define time {time() - start}")
#
# start = time()
# embedding_1 = gen_embedding(tf_model, old_batch.to(device))
#
#
# print (f"Model run time {time() - start}")
#
# start = time()
#
# loss = torch.sum(embedding_1)
#
#
# g = make_dot(loss)
# g.view()
#
# loss.backward()
#
# finish = time() - start
#
# print (f"Model backward time: {time() - start}")

#%%
#mem 7800 time 1.9
#%%

#%%

#%%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize
#%%
def review_preprocess(review):
    """
    Takes in a string of review, then performs the following:
    1. Remove HTML tag from review
    2. Remove URLs from review
    3. Make entire review lowercase
    4. Split the review in words
    5. Remove all punctuation
    6. Remove empty strings from review
    7. Remove all stopwords
    8. Returns a list of the cleaned review after jioning them back to a sentence
    """
    en_stops = set(stopwords.words('english'))

    """
    Removing HTML tag from review
    """
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', review)


    """
    Removing URLs
    """
    review_without_tag_and_url = re.sub(r"http\S+", "", review_without_tag)

    review_without_tag_and_url = re.sub(r"www\S+", "", review_without_tag)

    """
    Make entire string lowercase
    """
    review_lowercase = review_without_tag_and_url.lower()

    """
    Split string into words
    """
    list_of_words = word_tokenize(review_lowercase)


    """
    Remove punctuation
    Checking characters to see if they are in punctuation
    """

    list_of_words_without_punctuation=[''.join(this_char for this_char in this_string if (this_char in string.ascii_lowercase))for this_string in list_of_words]


    """
    Remove empty strings
    """
    list_of_words_without_punctuation = list(filter(None, list_of_words_without_punctuation))


    """
    Remove any stopwords
    """

    filtered_word_list = [w for w in list_of_words_without_punctuation if w not in en_stops]

    """
    Returns a list of the cleaned review after jioning them back to a sentence
    """
    return ' '.join(filtered_word_list)


"""
Load file into memory
"""
def load_file(filename):
    """
    Open the file as read only
    """
    file = open(filename, 'r')
    """
    Read all text
    """
    text = file.read()
    """
    Close the file
    """
    file.close()
    return text

def get_data(directory, vocab, is_trian):
    """
    Reading train test directory
    """
    review_dict={'neg':[],'pos':[]}
    if is_trian:
        directory = os.path.join(directory+'/train')
    else:
        directory = os.path.join(directory+'/test')
    print('Directory : ',directory)
    for label_type in ['neg', 'pos']:
            data_folder=os.path.join(directory, label_type)
            print('Data Folder : ',data_folder)
            for root, dirs, files in os.walk(data_folder):
                for fname in files:
                    if fname.endswith(".txt"):
                        file_name_with_full_path=os.path.join(root, fname)
                        review=load_file(file_name_with_full_path)
                        clean_review=review_preprocess(review)
                        if label_type == 'neg':
                            review_dict['neg'].append(clean_review)
                        else:
                            review_dict['pos'].append(clean_review)
                        """
                        Update counts
                        """
                        vocab.update(clean_review.split())

    return review_dict
#%%
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import time
import re
import string
#%%
startTime = time.time()
vocab = Counter()
directory='/home/sean/Downloads/aclImdb_v1/aclImdb'

try:
    with open("imbd_dicts", "rb") as f:
        train_review_dict, test_review_dict, word_list, vocab_to_int, int_to_vocab = pickle.load(f)
except:
    train_review_dict=get_data(directory, vocab, True)
    test_review_dict=get_data(directory, vocab, False)

    word_list = sorted(vocab, key = vocab.get, reverse = True)
    vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
    int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}

    with open("imbd_dicts", "wb") as f:
        pickle.dump((train_review_dict, test_review_dict, word_list, vocab_to_int, int_to_vocab), f)

total_time=time.time()-startTime

print('Time Taken : ',total_time/60,'minutes')
#%%
print('Number of negative reviews in train set :',len(train_review_dict['neg']))
print('Number of positive reviews in train set :',len(train_review_dict['pos']))

print('\nNumber of negative reviews in test set :',len(test_review_dict['neg']))
print('Number of positive reviews in test set :',len(test_review_dict['pos']))

#%%
#%%


class IMDBReviewDataset(Dataset):

    def __init__(self, review_dict, alphabet):


        self.data = review_dict
        self.labels = [x for x in review_dict.keys()]
        self.alphabet = alphabet

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        label = 0
        while idx >= len(self.data[self.labels[label]]):
            idx -= len(self.data[self.labels[label]])
            label += 1
        reviewText = self.data[self.labels[label]][idx]



        label_vec = torch.zeros((1), dtype=torch.long)
        label_vec[0] = label
        return self.reviewText2InputVec(reviewText), label

    def reviewText2InputVec(self, review_text):
        T = len(review_text)

        review_text_vec = torch.zeros((T), dtype=torch.long)
        encoded_review=[]
        for pos,word in enumerate(review_text.split()):
            if word not in vocab_to_int.keys():
                """
                If word is not available in vocab_to_int dict puting 0 in that place
                """
                review_text_vec[pos]=0
            else:
                review_text_vec[pos]=vocab_to_int[word]

        return review_text_vec



#%%
def pad_and_pack(batch):
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)

    longest = max(lengths)
    print (longest)
    print (sum(lengths) / len(lengths))
    #We need to pad all the inputs up to 'longest', and combine into a batch ourselves
    if len(input_tensors[0].shape) == 1:
        x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    else:
        raise Exception('Current implementation only supports (T) shaped data')

    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)

    y_batched = torch.as_tensor(labels, dtype=torch.long)

    return x_packed, y_batched
#%%
B = 24
train_dataset=IMDBReviewDataset(train_review_dict,vocab)
test_dataset=IMDBReviewDataset(test_review_dict,vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_and_pack)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_and_pack)

#%%
# for random batch, 3000 max len, 900 avg. So large padding difference
# batch = next(iter(train_loader))
#%%
# 831 avg length, 9100 max, so large difference
# sum([len(train_dataset[i][0]) for i in range(len(train_dataset))]) / len(train_dataset)
#%%
# train_dataset[0][0]

#%%
# [int_to_vocab[i] for i in train_dataset[0][0].tolist() if i != 0]
#%%
# len(vocab)
#%%
# train_dataset[0][0]
#%%
def vectorise_imdb(data, max_len = 129):
    X_G = [sample+([0]* (max_len-len(sample))) if len(sample)<max_len else sample[:max_len] for sample in data]
    return torch.tensor(X_G, dtype=torch.int32)

#%%
new_train = []
for x, y in train_dataset:
    x = [i for i in x.tolist() if i != 0]
    new_train.append((x,y))
#%%
# max([len(new_train[i][0]) for i in range(len(new_train))])
#%%
# import matplotlib.pyplot as plt

#%%
# plt.hist([len(new_train[i][0]) for i in range(len(new_train))])

#%%


#%%
def to_pyg_data(data_list):
    return [Data(x=torch.LongTensor(x[0]), y=torch.tensor(x[1])) if len(x[0]) < 128 else Data(x=torch.LongTensor(x[0][:128]), y=torch.tensor(x[1])) for x in data_list ]


train_graphs = to_pyg_data(new_train)


def to_tensor_list(data_list):
    return [(torch.LongTensor(x[0]),torch.tensor(x[1])) if len(x[0]) < 128 else (torch.LongTensor(x[0][:128]), torch.tensor(x[1])) for x in data_list ]


tensor_list = to_tensor_list(new_train)



#%%
# graph_loader = iter(gloader(final_graphs, batch_size=batch_size))
# batch = next(loader)
# batch.complete_edge_index = ptr_to_complete_edge_index(batch.ptr)
# pe = MPPositionalEncoding(embedding_dim)
# batch.pe = pe(batch.x, batch.ptr).squeeze(1)

#%%
vec_train = vectorise_imdb([new_train[i][0] for i in range(len(new_train))])

#%%
# new_train[0]
#%%
# vec_train[0]
#%%
# assert len(new_train) == len(vec_train)
#%%
final_train = [(vec_train[i], new_train[i][1]) for i in range(len(new_train))]
#%%
# new_train[0][0][:10]
#%%
# final_train[0][0][:10]
#%%
# loader = DataLoader(final_train, batch_size=32, shuffle=True)
#%%
# batch = next(iter(loader))
#%%
# batch
#%%
from models import inner_embedding_network

def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


def accuracy_transformer(x,y,model,fc):

    embedding = gen_embedding(model, x.to(device))

    preds = fc(x)

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(y).to(device)) / len(y)

def run_transformer_pretrain(step_size, decay_rate, num_epochs, batch_size, embedding_dim, save=False):

    # loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    # val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    loader = DataLoader(final_train, batch_size=batch_size, shuffle=True)


    # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TransformerEmbedding(ntoken=len(vocab), d_model=embedding_dim, nhead=1, d_hid=embedding_dim, nlayers=1).to(device)

    fc = inner_embedding_network.F_c_module_(embedding_dim).to(device)

    op_1 =torch.optim.Adam(model.parameters(), lr=step_size)

    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    # val_losses = []
    best_acc = 0.


    for j in range(num_epochs):
        print (f"Epoch: {j}")

        start = time.time()
        for batch_idx, (x,y) in enumerate(loader):
            # op_enc.zero_grad()
            op_1.zero_grad()
            op_fc.zero_grad()

            embedding = gen_embedding(model, x.to(device))

            preds = fc(embedding)

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), torch.LongTensor(y).to(device))

            loss.backward()

            op_1.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)

            if batch_idx % 100 == 0:

                # validation_loss = accuracy_transformer(model_1, model_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
                #
                # val_losses.append((validation_loss.detach(), j, batch_idx))
                #
                # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                # print ("Val acc: {}".format(validation_loss.detach()))

        print (f"Time for epoch {time.time() - start}")
    return #training_losses


#%%
# run_transformer_pretrain(1e-4, 0, 5, 32, 128)
#%%

#%%

#%%
def accuracy_mp_transformer(x,y,model,fc):

    embedding = gen_embedding(model, x.to(device))

    preds = fc(x)

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(y).to(device)) / len(y)

def run_mp_transformer_pretrain(step_size, decay_rate, num_epochs, batch_size, embedding_dim):

    # loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    # val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    # loader = DataLoader(final_train, batch_size=batch_size, shuffle=True)

    graph_loader = gloader(train_graphs, batch_size=batch_size, shuffle=True)

    # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MPTransformer(in_size=len(vocab), d_model=embedding_dim, dim_feedforward=embedding_dim, num_layers=1,num_heads=1,in_embed=True,dropout=0.,max_seq_len=None,batch_norm=False,pe=False, global_pool='max').to(device)

    fc = inner_embedding_network.F_c_module_(embedding_dim).to(device)

    op_1 =torch.optim.Adam(model.parameters(), lr=step_size)

    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    # val_losses = []
    best_acc = 0.

    pe = MPPositionalEncoding(embedding_dim)

    for j in range(num_epochs):
        print (f"Epoch: {j}")

        start = time.time()

        for batch_idx, batch in enumerate(graph_loader):
            # op_enc.zero_grad()
            op_1.zero_grad()
            op_fc.zero_grad()

            batch.complete_edge_index = ptr_to_complete_edge_index(batch.ptr)
            # batch.pe = pe(batch.x, batch.ptr).squeeze(1)

            def embed():
                return model(batch.to(device))

            embedding = embed()

            preds = fc(embedding)

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), batch.y.to(device))


            # g = make_dot(loss)
            # g.show()

            loss.backward()

            op_1.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)

            if batch_idx % 100 == 0:


                # validation_loss = accuracy_transformer(model_1, model_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
                #
                # val_losses.append((validation_loss.detach(), j, batch_idx))
                #
                # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                # print ("Val acc: {}".format(validation_loss.detach()))

        print (f"Time for epoch {time.time() - start}")

    return #training_losses

#%%
from cProfile import run
# run('run_transformer_pretrain(1e-4, 0, 5, 32, 128)', sort='cumtime')

#%%

# run('run_mp_transformer_pretrain(1e-4, 0, 5, 32, 128)', sort='cumtime')


def accuracy_nested_transformer(x,y,model,fc):

    embedding = gen_embedding(model, x.to(device))

    preds = fc(x)

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(y).to(device)) / len(y)

def run_nested_transformer_pretrain(step_size, decay_rate, num_epochs, batch_size, embedding_dim):

    # loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    # val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    # loader = DataLoader(final_train, batch_size=batch_size, shuffle=True)

    graph_loader = gloader(train_graphs, batch_size=batch_size, shuffle=True)

    # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NestedTransformer(in_size=len(vocab), d_model=embedding_dim, dim_feedforward=embedding_dim, num_layers=1,num_heads=1,in_embed=True,dropout=0.,max_seq_len=None,batch_norm=False,pe=False, global_pool='max').to(device)

    fc = inner_embedding_network.F_c_module_(embedding_dim).to(device)

    op_1 =torch.optim.Adam(model.parameters(), lr=step_size)

    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    # val_losses = []
    best_acc = 0.

    pe = MPPositionalEncoding(embedding_dim)

    for j in range(num_epochs):
        print (f"Epoch: {j}")

        start = time.time()

        for i in range(0, len(tensor_list) // batch_size):

            op_1.zero_grad()
            op_fc.zero_grad()

            batch = tensor_list[batch_size * i: batch_size * (i + 1)]

            y = torch.stack([batch[j][1] for j in range(len(batch))])

            x = [batch[j][0] for j in range(len(batch))]

            # op_enc.zero_grad()

            embedding = model(x.to(device))

            preds = fc(embedding)

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), y.to(device))


            # g = make_dot(loss)
            # g.show()

            loss.backward()

            op_1.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)

            if i % 100 == 0:


                # validation_loss = accuracy_transformer(model_1, model_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
                #
                # val_losses.append((validation_loss.detach(), j, batch_idx))
                #
                # val_loader = iter(DataLoader(val_dataset, batch_size=batch_size))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                # print ("Val acc: {}".format(validation_loss.detach()))

        print (f"Time for epoch {time.time() - start}")

    return #training_losses
