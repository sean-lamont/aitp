import math
import traceback
from models.graph_transformers.SAT.sat.position_encoding import RWEncoding, LapEncoding
from models.graph_transformers.SAT.sat.models import GraphTransformer
import random
from models import gnn_edge_labels
from tqdm import tqdm
import wandb
import os
# from utils.viz_net_torch import make_dot
import multiprocessing as multiprocessing
import cProfile
import torch
import torch_geometric.utils
from tqdm import tqdm
from models.digae_layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder, SingleLayerDirectedGCNConvEncoder
from models.gnn.digae.digae_model import OneHotDirectedGAE
import json
import models.gnn.formula_net.inner_embedding_network
from torch_geometric.data import Data
import pickle
# from data.hol4.ast_def import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
# from torchsummary import summary


# Data class for combining two graphs into one for supervised learning (i.e. premise selection). num_nodes_g1 tells us where in data.x to index to get nodes from g1
class CombinedGraphData(Data):
    def __init__(self, combined_x, combined_edge_index, num_nodes_g1, y, combined_edge_attr=None, complete_edge_index=None, pos_enc=None):
        super().__init__()
        self.y = y
        # node features concatenated along first dimension
        self.x = combined_x
        # adjacency matrix representing nodes from both graphs. Nodes from second graph have num_nodes_g1 added so they represent disjoint sets, but can be computed in parallel
        self.edge_index = combined_edge_index
        self.num_nodes_g1 = num_nodes_g1

        # combined edge features in format as above
        self.combined_edge_attr = combined_edge_attr

        self.complete_edge_index = complete_edge_index

        self.pos_enc = pos_enc

class LinkData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, edge_attr_s=None, edge_attr_t=None,
                 y=None, x_s_one_hot=None, x_t_one_hot=None, edge_index_s_complete=None, edge_index_t_complete=None,
                 depth_x_s=None, depth_x_t=None):
        super().__init__()

        self.edge_index_s = edge_index_s
        self.x_s = x_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t

        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t

        self.x_s_one_hot = x_s_one_hot
        self.x_t_one_hot = x_t_one_hot

        self.edge_index_t_complete = edge_index_t_complete
        self.edge_index_s_complete = edge_index_s_complete

        self.depth_x_s = depth_x_s
        self.depth_x_t = depth_x_t

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':# or key == 'edge_index_s_complete':
            return self.x_s.size(0)
        if key == 'edge_index_t':#or key == 'edge_index_t_complete':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def get_directed_edge_index(num_nodes, edge_idx):
    from_idx = []
    to_idx = []

    for i in range(0, num_nodes - 1):
        # to_idx = [i]
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx)
        except:
            print(f"Exception {i, num_nodes, edge_idx}")

        # ancestor_nodes = ancestor_nodes.item()
        found_nodes = list(ancestor_nodes).remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                              edge_index=edge_idx,
                                                                              flow='target_to_source')
        # children_nodes = children_nodes.item()
        # print (found_nodes, children_nodes, i, self_idx.item(), edge_idx)
        found_nodes = list(children_nodes).remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


# probably slow, could recursively do k-hop subgraph with k = 1 instead
def get_depth_from_graph(num_nodes, edge_index):
    from_idx = edge_index[0]
    to_idx = edge_index[1]

    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    # print (source_node)
    # prev_depth_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=0, edge_index=edge_index, flow='target_to_source')

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i,
                                                                          edge_index=edge_index,
                                                                          flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes

    return depths


vocab_size = 1909

# first update graph db to have directed edges + depth


# try:
#     with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/full_expr_dict.pk", "rb") as f:
#         graph_dict = pickle.load(f)
#     print("Full graph db loaded")
# except:
#     print("Generating full database with ancestor/children edges and depth information")
#     with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/global_expr_dict.pk", "rb") as f:
#         graph_db = pickle.load(f)
#
#     num_items = len(graph_db)
#
#     num_cores = multiprocessing.cpu_count()
#
#     items_per_core = (num_items // num_cores) + 1
#
#     inds_per_core = list(range(0, num_items, items_per_core))
#
#     inds_per_core.append(num_items - 1)
#
#     manager = multiprocessing.Manager()
#
#     global_dict = manager.dict(graph_db)
#
#     import random
#
#     keys = list(graph_db)
#
#     random.shuffle(keys)
#
#
#     def update_dict(from_idx, to_idx):
#         for i in tqdm(range(from_idx, to_idx)):
#             key = keys[i]
#             x, edge_index = global_dict[key]
#
#             num_nodes = len(x)
#
#             edge_index = torch.LongTensor(edge_index)
#
#             complete_edge_index = get_directed_edge_index(num_nodes, edge_index)
#
#             depth = get_depth_from_graph(num_nodes, edge_index)
#
#             global_dict[key] = (x, edge_index.tolist(), complete_edge_index.tolist(), depth.tolist())
#             # new_dict[key] = (x, edge_index.tolist(), complete_edge_index.tolist(), depth.tolist())
#
#
#     processes = []
#     for i in range(num_cores):
#         p = multiprocessing.Process(target=update_dict, args=(inds_per_core[i], inds_per_core[i + 1]))
#         processes.append(p)
#         p.start()
#
#     for process in processes:
#         process.join()
#
#     with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/full_expr_dict.pk", "wb") as f:
#         pickle.dump(dict(global_dict), f)
#
#     graph_dict = dict(global_dict)

# exit()


with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/train_data.pk", "rb") as f:
    train_data = pickle.load(f)

with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/val_data.pk", "rb") as f:
    val_data = pickle.load(f)

with open("/home/sean/Documents/phd/holist/holstep_gnn/formula_net/data/graph_data/data.pk", "rb") as f:
    test_data = pickle.load(f)


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


def loss(graph_net, batch, fc):  # , F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):

    g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))

    g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    eps = 1e-6

    preds = torch.clip(preds, eps, 1 - eps)

    return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def positional_encoding(d_model, depth_vec):
    size, _ = depth_vec.shape

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(size, d_model)

    pe[:, 0::2] = torch.sin(depth_vec * div_term)
    pe[:, 1::2] = torch.cos(depth_vec * div_term)

    return pe

vocab_size = 1909

def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


def to_combined_batch(list_data, data_dict, embedding_dim):
    batch_list = []
    for (y, conj,stmt) in list_data:
        # x1/x_t is conj, x2/x_s is stmt

        x, edge_index, _, depth = data_dict[conj]

        x1_mat = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=1909)  # .float()

        x1_edge_index = torch.LongTensor(edge_index)

        num_nodes_g1 = x1_mat.shape[0]#conj_graph.num_nodes

        x, edge_index, _, depth = data_dict[stmt]

        x2_mat = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=1909)  # .float()

        edge_index2 = torch.LongTensor(edge_index) + num_nodes_g1

        # batch_list.append(
        #     LinkData(edge_index_s=stmt_graph.edge_index, x_s=stmt_graph.x, edge_attr_s=stmt_graph.edge_attr, edge_index_t=conj_graph.edge_index, x_t=conj_graph.x, edge_attr_t=conj_graph.edge_attr, y=torch.tensor(y)))

        # Concatenate node feature matrices
        combined_features = torch.cat([x1_mat, x2_mat], dim=0)

        # Combine edge indices




        combined_edge_index = torch.cat([x1_edge_index, edge_index2], dim=1)

        # combine edge attributes
        # edge_attr1 = conj_graph.edge_attr
        # edge_attr2 = stmt_graph.edge_attr
        # combined_edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)


        # Compute disjoint pairwise complete edge indices
        complete_edge_index1 = torch.cartesian_prod(torch.arange(num_nodes_g1),
                                                    torch.arange(num_nodes_g1))  # All pairs of nodes in conj_graph

        complete_edge_index2 = torch.cartesian_prod(torch.arange(num_nodes_g1, num_nodes_g1 + x2_mat.shape[0]),
                                                    torch.arange(num_nodes_g1,
                                                                 num_nodes_g1 + x2_mat.shape[0]))  # All pairs of nodes in stmt_graph

        complete_edge_index = torch.cat([complete_edge_index1, complete_edge_index2], dim=0).t().contiguous()


        # todo Note, if we instead took the complete edge index of the combined graphs, the node embeddings would attend to all nodes in both graphs.
        # doing it this way, we are doing normal graph_benchmarks for each graph, then the CLS node will take a global attention over the intra graph node embeddings

        # positional encodings

        graph_ind = torch.cat([torch.ones(num_nodes_g1), torch.ones(x2_mat.shape[0]) * 2], dim=0)
        pos_enc = positional_encoding(embedding_dim, graph_ind.unsqueeze(1))

        #append combined graph to batch

        # batch_list.append(CombinedGraphData(combined_x=combined_features,combined_edge_index=combined_edge_index, combined_edge_attr=combined_edge_attr, complete_edge_index=complete_edge_index, num_nodes_g1=num_nodes_g1, pos_enc=pos_enc, y=y))
        batch_list.append(CombinedGraphData(combined_x=combined_features,combined_edge_index=combined_edge_index,  complete_edge_index=complete_edge_index, num_nodes_g1=num_nodes_g1, pos_enc=pos_enc, y=y))


    loader = iter(DataLoader(batch_list, batch_size=len(batch_list)))

    batch = next(iter(loader))

    return batch




# run_digae(1e-4, 0, 200, 128, 256, 2, save=false)

def get_model(config):

    if config['model_type'] == 'graph_benchmarks':
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
                                dropout=config['dropout'],
                                k_hop=config['gnn_layers'],
                                global_pool=config['global_pool'])

    elif config['model_type'] == 'formula-net':
        return models.gnn.formula_net.inner_embedding_network.FormulaNet(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'digae':
        return None

    elif config['model_type'] == 'classifier':
        return None

    else:
        return None


# def run_dual_encoders(model_config, exp_config):
def run_combined_graphs(config):

    model_config = config['model_config']
    exp_config = config['exp_config']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_net_combined = get_model(model_config).to(device)

    print ("Model details:")

    print(graph_net_combined)

    wandb.log({"Num_model_params": sum([p.numel() for p in graph_net_combined.parameters() if p.requires_grad])})

    embedding_dim = model_config['embedding_dim']
    lr = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    save = exp_config['model_save']
    val_size = exp_config['val_size']


    #wandb load
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']

    fc = gnn_edge_labels.BinaryClassifier(embedding_dim).to(device)


    op_g1 = torch.optim.AdamW(graph_net_combined.parameters(), lr=lr, weight_decay=weight_decay)
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
                batch = to_combined_batch(train_data[from_idx:to_idx], graph_dict, embedding_dim)
            except Exception as e:
                # print(f"Error in batch {i}: {e}")
                # traceback.print_exc()
                continue

            # op_enc.zero_grad()
            op_g1.zero_grad()
            op_fc.zero_grad()

            # edge attributes
            # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_t_complete.to(device), abs_pe =positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_t.long().to(device))
            # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_s_complete.to(device), abs_pe = positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_s.long().to(device))

            data = Data(x=batch.x.float().to(device), edge_index=batch.edge_index.to(device),
                        batch=batch.batch.to(device),
                        ptr=batch.ptr.to(device),
                        complete_edge_index=batch.complete_edge_index.to(device),
                        abs_pe=batch.pos_enc.to(device))#,

            try:
                graph_enc_1 = graph_net_combined(data)

                preds = fc(graph_enc_1)

                eps = 1e-6

                preds = torch.clip(preds, eps, 1 - eps)

                loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))


                loss.backward()

                # op_enc.step()
                op_g1.step()
                op_fc.step()


            except Exception as e:
                err_count += 1
                if err_count > 100:
                    return Exception("Too many errors in training")
                # traceback.print_exc()
                # print(f"Error in training {e}")
                continue

            training_losses.append(loss.detach() / batch_size)

            if i % (100000 // batch_size) == 0:

                graph_net_combined.eval()

                val_count = []

                random.shuffle(val_data)


                val_inds = list(range(0, len(val_data), batch_size))

                val_inds.append(len(val_data) - 1)

                for k in tqdm(range(0, val_size  // batch_size)):
                    val_err_count = 0

                    from_idx_val = val_inds[k]
                    to_idx_val = val_inds[k + 1]

                    try:
                        val_batch = to_combined_batch(val_data[from_idx_val:to_idx_val], graph_dict, embedding_dim)
                        validation_loss = val_acc_dual_encoder(graph_net_combined, val_batch,
                                                               fc, embedding_dim)  # , fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                        val_count.append(validation_loss.detach())
                    except Exception as e:
                        # print(f"Error {e}, batch: {val_batch}")
                        # traceback.print_exc()
                        val_err_count += 1
                        continue

                print (f"Val errors: {val_err_count}")

                validation_loss = (sum(val_count) / len(val_count)).detach()
                val_losses.append((validation_loss, j, i))

                print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print("Val acc: {}".format(validation_loss.detach()))

                print (f"Failed batches: {err_count}")

                wandb.log({"acc": validation_loss.detach(),
                           "train_loss_avg": sum(training_losses[-100:]) / len(training_losses[-100:]),
                           "epoch": j})

                if validation_loss > best_acc:
                    best_acc = validation_loss
                    print(f"New best validation accuracy: {best_acc}")
                    # only save encoder if best accuracy so far
                    if save == True:
                        torch.save(graph_net_combined, "sat_encoder_goal")

                    # wandb save
                    # torch.save({  # Save our checkpoint loc
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss,
                    # }, CHECKPOINT_PATH)
                    # wandb.save(CHECKPOINT_PATH)  # saves c


                graph_net_combined.train()

        wandb.log({"failed_batches": err_count})
    print(f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses


def val_acc_dual_encoder(model_1,  batch, fc, embedding_dim):

    data = Data(x=batch.x.float().to(device), edge_index=batch.edge_index.to(device),
                batch=batch.batch.to(device),
                ptr=batch.ptr.to(device),
                complete_edge_index=batch.complete_edge_index.to(device),
                abs_pe=batch.pos_enc.to(device))#,

    graph_enc_1 = model_1(data)

    preds = fc(graph_enc_1)

    preds = torch.flatten(preds)

    preds = (preds > 0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


# data_config = {

#
# }


sat_config = {
    "model_type": "graph_benchmarks",
    "vocab_size": 1909,
    "embedding_dim": 128,
    "dim_feedforward": 256,
    "num_heads": 4,
    "num_layers": 4,
    "in_embed": False,
    "se": "pna",
    "abs_pe": False,
    "abs_pe_dim": 128,
    "use_edge_attr": False,
    "dropout": 0.2,
    "gnn_layers": 4,
    "global_pool": "mean",
}

exp_config = {
    "learning_rate": 1e-4,
    "epochs": 5,
    "weight_decay": 1e-6,
    "batch_size": 128,
    "model_save": False,
    "val_size": 10000,
}

formula_net_config = {
    "model_type": "formula-net",
    "vocab_size": 1909,
    "embedding_dim": 256,
    "gnn_layers": 4,
}


#initialise with default parameters

# run_dual_encoders(run.config)

def main():
    wandb.init(
        project="test_project",

        name="Large Model",
        # track model and experiment configurations
        config={
            "exp_config": exp_config,
            "model_config": sat_config,
        }
    )

    wandb.define_metric("acc", summary="max")

    run_combined_graphs(wandb.config)

    return

# run_combined_graphs(
# config={
#     "exp_config": exp_config,
#     "model_config": sat_config,
# })
# main()

# exp_config['epochs'] = 2
# exp_config['model_save'] = False
#
# sweep_configuration = {
#     "method": "bayes",
#     "metric": {'goal': 'maximize', 'name': 'acc'},
#     "parameters": {
#         "model_config" : {
#             "parameters": {
#                 "model_type": {"values":["graph_benchmarks"]},
#                 "vocab_size": {"values":[1909]},
#                 "embedding_dim": {"values":[128, 256]},
#                 "in_embed": {"values":[False]},
#                 "abs_pe": {"values":[False]},
#                 "abs_pe_dim": {"values":[None]},
#                 "use_edge_attr": {"values":[False]},
#                 "dim_feedforward": {"values": [256]},
#                 "num_heads": {"values": [8]},
#                 "num_layers": {"values": [4]},
#                 "se": {"values": ['pna']},
#                 "dropout": {"values": [0.2]},
#                 "gnn_layers": {"values": [0, 2, 4]},
#                 "global_pool": {"values": ['cls']}
#                 }
#             }
#         }
#     }
#
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='hol4_premise_selection')
#
# wandb.agent(sweep_id,function=main)
# # #
# # #
# # #
# # #
# # #
