import traceback

from torch import nn
import wandb
import random
import cProfile
import models.gnn_edge_labels
from models.graph_transformers.SAT.sat.models import GraphTransformer, AMRTransformer
import math
import torch
import torch_geometric.utils
from tqdm import tqdm
from models.digae_layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder, SingleLayerDirectedGCNConvEncoder
from models.digae_model import OneHotDirectedGAE
import json
import models.inner_embedding_network
from torch_geometric.data import Data
import pickle
from data.hol4.ast_def import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

data_dir = "data/hol4/data/"
data_dir = os.path.join(os.getcwd(),data_dir)

with open(data_dir + "dep_data.json") as fp:
    dep_db = json.load(fp)
    
with open(data_dir + "adjusted_db.json") as fp:
    new_db = json.load(fp)

with open(data_dir + "torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

with open(data_dir + "train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)

polished_goals = []

for val_ in new_db.values():
    polished_goals.append(val_[2])

tokens = list(set([token.value for polished_goal in polished_goals for token in polished_to_tokens_2(polished_goal)  if token.value[0] != 'V']))

tokens.append("VAR")
tokens.append("VARFUNC")
tokens.append("UNKNOWN")


class LinkData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, edge_attr_s=None, edge_attr_t = None,
                 y=None, x_s_one_hot=None, x_t_one_hot=None, edge_index_s_complete=None, edge_index_t_complete=None, depth_x_s=None, depth_x_t=None):
        super().__init__()

        self.edge_index_s = edge_index_s
        self.x_s = x_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t

        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t

        self.x_s_one_hot=x_s_one_hot
        self.x_t_one_hot=x_t_one_hot

        self.edge_index_t_complete = edge_index_t_complete
        self.edge_index_s_complete = edge_index_s_complete

        self.depth_x_s = depth_x_s
        self.depth_x_t = depth_x_t


        if edge_index_s is not None and edge_index_t is not None:
            self.softmax_idx_s = self.edge_index_s.size(1)
            self.softmax_idx_t = self.edge_index_t.size(1)

        else:
            self.softmax_idx_s, self.edge_index_t = None, None

        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s' or key == 'edge_index_s_complete':
            return self.x_s.size(0)
        elif key == 'edge_index_t' or key == 'edge_index_t_complete':
            return self.x_t.size(0)

        elif key == 'softmax_idx_s':
            return self.softmax_idx_s
        elif key == 'softmax_idx_t':
            return self.softmax_idx_t

        return super().__inc__(key, value, *args, **kwargs)


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

#todo remove this when cartesian product used
def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


def to_amr_batch(list_data, data_dict):

    batch_list = []

    for (x1, x2, y) in list_data:
        # x1/x_t is conj, x2/x_s is stmt
        conj = x1
        stmt = x2

        conj_graph = data_dict[conj]
        stmt_graph = data_dict[stmt]

        batch_list.append(LinkData(edge_index_s=stmt_graph.edge_index, x_s=stmt_graph.x,
                                   edge_attr_s=stmt_graph.edge_attr.long(),  edge_index_t=conj_graph.edge_index,
                                   x_t=conj_graph.x, edge_attr_t=conj_graph.edge_attr.long(), y=torch.tensor(y)))




    loader = iter(DataLoader(batch_list, batch_size=len(batch_list), follow_batch=['x_s', 'x_t']))

    batch = next(iter(loader))

    return batch


def get_model(config):

    if config['model_type'] == 'sat':
        return GraphTransformer(in_size=config['vocab_size'],
                                num_class=2,
                                d_model=config['embedding_dim'],
                                dim_feedforward=config['dim_feedforward'],
                                num_heads=config['num_heads'],
                                num_layers=config['num_layers'],
                                in_embed=config['in_embed'],
                                se=config['se'],
                                abs_pe=config['abs_pe'],
                                abs_pe_dim=config['abs_pe_dim'],
                                use_edge_attr=config['use_edge_attr'],
                                num_edge_features=200,
                                dropout=config['dropout'],
                                k_hop=config['gnn_layers'])

    if config['model_type'] == 'amr':
        return AMRTransformer(in_size=config['vocab_size'],
                                d_model=config['embedding_dim'],
                                dim_feedforward=config['dim_feedforward'],
                                num_heads=config['num_heads'],
                                num_layers=config['num_layers'],
                                in_embed=config['in_embed'],
                                abs_pe=config['abs_pe'],
                                abs_pe_dim=config['abs_pe_dim'],
                                use_edge_attr=config['use_edge_attr'],
                                num_edge_features=200,
                                dropout=config['dropout'],
                                layer_norm=True,
                                global_pool='cls',
                                )

    elif config['model_type'] == 'formula-net':
        return models.inner_embedding_network.FormulaNet(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'formula-net-edges':
        return models.gnn_edge_labels.message_passing_gnn_edges(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'digae':
        return None

    elif config['model_type'] == 'classifier':
        return None

    else:
        return None


train_data = train
val_data = val


def run_dual_encoders(config):

    model_config = config['model_config']
    exp_config = config['exp_config']

    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_net_1 = get_model(model_config).to(device)
    graph_net_2 = get_model(model_config).to(device)
    print ("Model details:")

    print(graph_net_1)


    embedding_dim = model_config['embedding_dim']
    lr = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    save = exp_config['model_save']
    val_size = exp_config['val_size']
    logging = exp_config['logging']

    if 'directed_attention' in model_config:
        directed_attention = model_config['directed_attention']
    else:
        directed_attention = False

    if logging:
        wandb.log({"Num_model_params": sum([p.numel() for p in graph_net_1.parameters() if p.requires_grad])})

    #wandb load
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']

    fc = models.gnn_edge_labels.F_c_module_(embedding_dim * 4).to(device)
    # fc = nn.DataParallel(fc).to(device)
    # fc = torch.compile(fc)

    op_g1 = torch.optim.AdamW(graph_net_1.parameters(), lr=lr, weight_decay=weight_decay)
    op_g2 = torch.optim.AdamW(graph_net_2.parameters(), lr=lr, weight_decay=weight_decay)
    op_fc = torch.optim.AdamW(fc.parameters(), lr=lr, weight_decay=weight_decay)

    training_losses = []

    val_losses = []
    best_acc = 0.

    inds = list(range(0, len(train_data), batch_size))
    inds.append(len(train_data) - 1)

    random.shuffle(train_data)

    for j in range(epochs):
        print(f"Epoch: {j}")
        # for i, batch in tqdm(enumerate(loader)):
        err_count = 0
        for i in tqdm(range(0, len(inds) - 1)):

            from_idx = inds[i]
            to_idx = inds[i + 1]

            try:
                batch = to_amr_batch(train_data[from_idx:to_idx], torch_graph_dict)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                traceback.print_exc()
                continue

            # op_enc.zero_grad()
            op_g1.zero_grad()
            op_g2.zero_grad()
            op_fc.zero_grad()


            data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
                          batch=batch.x_t_batch.to(device),
                          # edge_index_source = torch.LongTensor([[i for i in torch.arange(batch.edge_index_t.shape[1])], [batch.edge_index_t[0][i] for i in torch.arange(batch.edge_index_t.shape[1])]]).to(device),
                          # edge_index_target = torch.LongTensor([[i for i in torch.arange(batch.edge_index_t.shape[1])], [batch.edge_index_t[1][i] for i in torch.arange(batch.edge_index_t.shape[1])]]).to(device),
                          abs_pe = torch.ones(batch.x_t.size(0)).long().to(device),
                          softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_t]).to(device),
                          ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
                          edge_attr=batch.edge_attr_t.long().to(device))

            data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
                          batch=batch.x_s_batch.to(device),
                          # edge_index_source = torch.LongTensor([[i for i in torch.arange(batch.edge_index_s.shape[1])], [batch.edge_index_s[0][i] for i in torch.arange(batch.edge_index_s.shape[1])]]).to(device),
                          # edge_index_target = torch.LongTensor([[i for i in torch.arange(batch.edge_index_s.shape[1])], [batch.edge_index_s[1][i] for i in torch.arange(batch.edge_index_s.shape[1])]]).to(device),
                          ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
                          softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_s]).to(device),
                          abs_pe = torch.zeros(batch.x_s.size(0)).long().to(device),
                          # softmax_idx = batch.softmax_idx_s.to(device),
                          edge_attr=batch.edge_attr_s.long().to(device))

            # print (data_1.softmax_idx, data_1.ptr)

            try:
                graph_enc_1 = graph_net_1(data_1.to(device))

                # print (f"data1 {data_1}")

                graph_enc_2 = graph_net_2(data_2.to(device))

                preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

                eps = 1e-6

                preds = torch.clip(preds, eps, 1 - eps)

                loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))


                loss.backward()

                # op_enc.step()
                op_g1.step()
                op_g2.step()
                op_fc.step()


            except Exception as e:
                err_count += 1
                if err_count > 100:
                    return Exception("Too many errors in training")
                print(f"Error in training {e}")
                traceback.print_exc()
                continue

            training_losses.append(loss.detach() / batch_size)

            if i % (10000 // batch_size) == 0:

                graph_net_1.eval()
                graph_net_2.eval()

                val_count = []

                random.shuffle(val_data)

                val_inds = list(range(0, len(val_data), batch_size))
                val_inds.append(len(val_data) - 1)


                for k in tqdm(range(0, val_size // batch_size)):
                    val_err_count = 0

                    from_idx_val = val_inds[k]
                    to_idx_val = val_inds[k + 1]

                    try:
                        val_batch = to_amr_batch(val_data[from_idx_val:to_idx_val], torch_graph_dict)

                        validation_loss = val_acc_dual_encoder(graph_net_1, graph_net_2, val_batch,
                                                               fc, embedding_dim, directed_attention)  # , fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                        val_count.append(validation_loss.detach())
                    except Exception as e:
                        # print(f"Error {e}, batch: {val_batch}")
                        print (e)
                        val_err_count += 1
                        continue

                print (f"Val errors: {val_err_count}")

                validation_loss = (sum(val_count) / len(val_count)).detach()
                val_losses.append((validation_loss, j, i))

                print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print("Val acc: {}".format(validation_loss.detach()))

                print (f"Failed batches: {err_count}")

                if logging:
                    wandb.log({"acc": validation_loss.detach(),
                               "train_loss_avg": sum(training_losses[-100:]) / len(training_losses[-100:]),
                               "epoch": j})

                if validation_loss > best_acc:
                    best_acc = validation_loss
                    print(f"New best validation accuracy: {best_acc}")
                    # only save encoder if best accuracy so far

                    if save == True:
                        torch.save(graph_net_1, exp_config['model_dir'] + "/gnn_transformer_goal_hol4")
                        torch.save(graph_net_2, exp_config['model_dir'] + "/gnn_transformer_premise_hol4")

                    # wandb save
                    # torch.save({  # Save our checkpoint loc
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss,
                    # }, CHECKPOINT_PATH)
                    # wandb.save(CHECKPOINT_PATH)  # saves c


                graph_net_1.train()
                graph_net_2.train()

        if logging:
            wandb.log({"failed_batches": err_count})

    print(f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses


def val_acc_dual_encoder(model_1, model_2, batch, fc, embedding_dim, directed_attention):

    # data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
    #               batch=batch.x_t_batch.to(device),
    #               ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
    #               edge_attr=batch.edge_attr_t.long().to(device))
    #
    # data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
    #               batch=batch.x_s_batch.to(device),
    #               ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
    #               edge_attr=batch.edge_attr_s.long().to(device))


    data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
                  batch=batch.x_t_batch.to(device),
                  # edge_index_source = torch.LongTensor([[i for i in torch.arange(batch.edge_index_t.shape[1])], [batch.edge_index_t[0][i] for i in torch.arange(batch.edge_index_t.shape[1])]]).to(device),
                  # edge_index_target = torch.LongTensor([[i for i in torch.arange(batch.edge_index_t.shape[1])], [batch.edge_index_t[1][i] for i in torch.arange(batch.edge_index_t.shape[1])]]).to(device),
                  # softmax_idx = batch.softmax_idx_t.to(device),
                  softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_t]).to(device),
                  ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
                  edge_attr=batch.edge_attr_t.long().to(device))

    data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
                  batch=batch.x_s_batch.to(device),
                  # edge_index_source = torch.LongTensor([[i for i in torch.arange(batch.edge_index_s.shape[1])], [batch.edge_index_s[0][i] for i in torch.arange(batch.edge_index_s.shape[1])]]).to(device),
                  # edge_index_target = torch.LongTensor([[i for i in torch.arange(batch.edge_index_s.shape[1])], [batch.edge_index_s[1][i] for i in torch.arange(batch.edge_index_s.shape[1])]]).to(device),
                  ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
                  softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_s]).to(device),
                  # softmax_idx = batch.softmax_idx_s.to(device),
                  edge_attr=batch.edge_attr_s.long().to(device))


    graph_enc_1 = model_1(data_1)

    graph_enc_2 = model_2(data_2)

    preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds > 0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)

amr_config = {
    "model_type": "amr",
    "vocab_size": len(tokens),
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "in_embed": False,
    "abs_pe": False,
    "abs_pe_dim":2,
    "use_edge_attr": True,
    "dropout": 0.2,
}

exp_config = {
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "model_save": False,
    "val_size": 2048,
    "logging": False,
    "model_dir": "/home/sean/Documents/phd/aitp/experiments/hol4/supervised/model_checkpoints"
}

formula_net_config = {
    "model_type": "formula-net",
    "vocab_size": len(tokens),
    "embedding_dim": 256,
    "gnn_layers": 4,
}


def main():
    wandb.init(
        project="hol4_premise_selection",

        name="Directed Attention Sweep Separate Encoder",
        # track model and experiment configurations
        config={
            "exp_config": exp_config,
            "model_config": amr_config,
        }
    )

    wandb.define_metric("acc", summary="max")

    run_dual_encoders(wandb.config)

    return


# import cProfile
# cProfile.run('run_dual_encoders(config = {"model_config": amr_config, "exp_config": exp_config})', sort='cumtime')

run_dual_encoders(config = {"model_config": amr_config, "exp_config": exp_config})
# sweep_configuration = {
#     "method": "bayes",
#     "metric": {'goal': 'maximize', 'name': 'acc'},
#     "parameters": {
#         "model_config" : {
#             "parameters": {
#                 "model_type": {"values":["sat"]},
#                 "vocab_size": {"values":[len(tokens)]},
#                 "embedding_dim": {"values":[128]},
#                 "in_embed": {"values":[False]},
#                 "abs_pe": {"values":[True, False]},
#                 "abs_pe_dim": {"values":[128]},
#                 "use_edge_attr": {"values":[True, False]},
#                 "dim_feedforward": {"values": [256]},
#                 "num_heads": {"values": [8]},
#                 "num_layers": {"values": [4]},
#                 "se": {"values": ['pna']},
#                 "dropout": {"values": [0.2]},
#                 "gnn_layers": {"values": [0,4]},
#                 "directed_attention": {"values": [True,False]}
#             }
#         }
#     }
# }
#
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='hol4_premise_selection')
# #
# wandb.agent(sweep_id,function=main)
#
