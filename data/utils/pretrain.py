from torch_geometric.data import Data
from utils.mongodb_utils import get_batches
from tqdm import tqdm
import traceback
import wandb
from pymongo import MongoClient
from models import gnn_edge_labels, inner_embedding_network
from models.graph_transformers.SAT.sat.models import GraphTransformer, AMRTransformer
from torch_geometric.loader import DataLoader
import torch


# todo Better way for softmax_idx, combined graph setup, positonal encoding, model saving/versioning, parametrised classifier, DAGLSTM?

class LinkData(Data):
    def __init__(self, edge_index_s=None,
                 x_s=None, edge_index_t=None,
                 x_t=None, edge_attr_s=None,
                 edge_attr_t=None,
                 softmax_idx_s=None,
                 softmax_idx_t=None,
                 y=None):

        super().__init__()

        self.edge_index_s = edge_index_s
        self.x_s = x_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t

        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t

        # softmax index used for AMR model
        self.softmax_idx_s = softmax_idx_s
        self.softmax_idx_t = softmax_idx_t

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s' or key == 'attention_edge_index_s':
            return self.x_s.size(0)

        elif key == 'edge_index_t' or key == 'attention_edge_index_t':
            return self.x_t.size(0)

        elif key == 'softmax_idx_s':
            return self.softmax_idx_s

        elif key == 'softmax_idx_t':
            return self.softmax_idx_t

        return super().__inc__(key, value, *args, **kwargs)

def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

# redundant
def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index

def to_batch_mongo(batch, graph_collection, options):
    batch_list = []


    stmts = list(set([sample['stmt'] for sample in batch]))
    conjs = list(set([sample['conj'] for sample in batch]))

    stmts.extend(conjs)

    exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))

    expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}


    for sample in batch:

        stmt = sample['stmt']
        conj = sample['conj']
        y = sample['y']

        conj_graph = expr_dict[conj]
        stmt_graph = expr_dict[stmt]

        # todo compare speed with temporary dict vs fetching each item

        # conj_graph = graph_collection.find_one({"_id": conj})['graph']
        # stmt_graph = graph_collection.find_one({"_id": stmt})['graph']

        x1 = conj_graph['onehot']
        x1_mat = torch.LongTensor(x1)

        x2 = stmt_graph['onehot']
        x2_mat = torch.LongTensor(x2)

        tmp_batch = LinkData(x_s=x2_mat, x_t=x1_mat, y=torch.tensor(y))


        if 'edge_index' in options:
            if 'edge_index' in conj_graph and 'edge_index' in stmt_graph:
                x1_edge_index = conj_graph['edge_index']
                x1_edge_index = torch.LongTensor(x1_edge_index)

                x2_edge_index = stmt_graph['edge_index']
                x2_edge_index = torch.LongTensor(x2_edge_index)

                tmp_batch.edge_index_t = x1_edge_index
                tmp_batch.edge_index_s = x2_edge_index
            else:
                raise NotImplementedError


        if 'edge_attr' in options:
            if 'edge_attr' in conj_graph and 'edge_attr' in stmt_graph:
                x1_edge_attr = conj_graph['edge_attr']
                x1_edge_attr = torch.LongTensor(x1_edge_attr)

                x2_edge_attr = stmt_graph['edge_attr']
                x2_edge_attr = torch.LongTensor(x2_edge_attr)

                tmp_batch.edge_attr_t = x1_edge_attr
                tmp_batch.edge_attr_s = x2_edge_attr
            else:
                raise NotImplementedError

        # Edge index used to determine where attention is propagated in Message Passing Attention schemes

        if 'attention_edge_index' in options:
            if 'attention_edge_index' in conj_graph and 'attention_edge_index' in stmt_graph:
                tmp_batch.attention_edge_index_t = conj_graph['attention_edge_index']
                tmp_batch.attention_edge_index_s = stmt_graph['attention_edge_index']
            else:
                # Default is global attention
                tmp_batch.attention_edge_index_t = torch.cartesian_prod(torch.arange(x1_mat.size(0)),
                                                    torch.arange(x1_mat.size(0))).transpose(0,1)

                tmp_batch.attention_edge_index_s = torch.cartesian_prod(torch.arange(x2_mat.size(0)),
                                                    torch.arange(x2_mat.size(0))).transpose(0,1)


        #todo make data options have possible values i.e. options['softmax_idx'] == AMR, use edges, else directed attention etc.
        if 'softmax_idx' in options:
            tmp_batch.softmax_idx_t = x1_edge_index.size(1)
            tmp_batch.softmax_idx_s = x2_edge_index.size(1)


        # todo positional encoding including with depth


        batch_list.append(tmp_batch)


    loader = iter(DataLoader(batch_list, batch_size=len(batch_list), follow_batch=['x_s', 'x_t']))

    batch_ = next(iter(loader))

    return batch_

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
                              device=config['device']
                              )

    elif config['model_type'] == 'formula-net':
        return inner_embedding_network.FormulaNet(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'formula-net-edges':
        return gnn_edge_labels.message_passing_gnn_edges(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'digae':
        return None

    elif config['model_type'] == 'classifier':
        return None

    else:
        return None

def get_data(config):
    if config['data_source'] == "MongoDB":
        client = MongoClient()
        db = client[config['dbname']]
        graph_collection = db[config['graph_collection_name']]
        split_collection = db[config['split_name']]
        return graph_collection, split_collection
    else:
        return NotImplementedError


def separate_link_batch(batch, options):
    # assume data will always have at least x variable
    data_1 = Data(x=batch.x_t, batch=batch.x_t_batch, ptr=batch.x_t_ptr)
    data_2 = Data(x=batch.x_s, batch=batch.x_s_batch, ptr=batch.x_s_ptr)

    if 'edge_index' in options:
        data_1.edge_index = batch.edge_index_t
        data_2.edge_index = batch.edge_index_s

    if 'softmax_idx' in options:
        data_1.softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_t])
        data_2.softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_s])

    if 'edge_attr' in options:
        data_1.edge_attr = batch.edge_attr_t.long()
        data_2.edge_attr = batch.edge_attr_s.long()

    if 'attention_edge_index' in options:
        data_1.attention_edge_index = batch.attention_edge_index_t
        data_2.attention_edge_index = batch.attention_edge_index_s

    # todo
    # if 'abs_pe' in options:

    return data_1, data_2


def run_dual_encoders(config):
    model_config = config['model_config']
    exp_config = config['exp_config']
    data_options = config['data_config']['data_options']
    source_config = config['data_config']['source_config']

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    embedding_dim = model_config['embedding_dim']
    lr = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    save = exp_config['model_save']
    val_size = exp_config['val_size']
    logging = exp_config['logging']
    device = exp_config['device']
    max_errors = exp_config['max_errors']
    val_frequency = exp_config['val_frequency']

    graph_net_1 = get_model(model_config).to(device)
    graph_net_2 = get_model(model_config).to(device)

    print("Model details:")

    print(graph_net_1)

    if logging:
        wandb.log({"Num_model_params": sum([p.numel() for p in graph_net_1.parameters() if p.requires_grad])})

    fc = gnn_edge_labels.F_c_module_(embedding_dim * 4).to(device)

    op_g1 = torch.optim.AdamW(graph_net_1.parameters(), lr=lr, weight_decay=weight_decay)

    op_g2 = torch.optim.AdamW(graph_net_2.parameters(), lr=lr, weight_decay=weight_decay)

    op_fc = torch.optim.AdamW(fc.parameters(), lr=lr, weight_decay=weight_decay)

    training_losses = []

    val_losses = []
    best_acc = 0.

    graph_collection, split_collection = get_data(source_config)

    # train_cursor = split_collection.find({"split":"train"})

    # val_cursor = split_collection.aggregate([{"$match": {"split": "valid"}}, {"$sample": {"size": 10000000}}])
    val_cursor = split_collection.find({"split":"valid"}).limit(val_size)

    for j in range(epochs):
        print(f"Epoch: {j}")
        err_count = 0

        # train_cursor.rewind()

        train_cursor = split_collection.aggregate([{"$match": {"split": "train"}}, {"$sample": {"size": 10000000}}])
        batches = get_batches(train_cursor, batch_size)

        for i,db_batch in tqdm(enumerate(batches)):

            try:
                batch = to_batch_mongo(db_batch, graph_collection, data_options)
            except Exception as e:
                print(f"Error in batch: {e}")
                traceback.print_exc()
                continue

            op_g1.zero_grad()
            op_g2.zero_grad()
            op_fc.zero_grad()

            data_1, data_2 = separate_link_batch(batch, data_options)

            # print (data_1.x)
            # print (torch.sum(data_1.x))
            # exit()


            try:
                graph_enc_1 = graph_net_1(data_1.to(device))

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
                if err_count > max_errors:
                    return Exception("Too many errors in training")
                print(f"Error in training {e}")
                traceback.print_exc()
                continue

            training_losses.append(loss.detach() / batch_size)

            if i % val_frequency == 0:

                graph_net_1.eval()
                graph_net_2.eval()

                val_count = []

                val_cursor.rewind()

                get_val_batches = get_batches(val_cursor, batch_size)

                for db_val in get_val_batches:
                    val_err_count = 0
                    try:
                        val_batch = to_batch_mongo(db_val, graph_collection, data_options)

                        validation_loss = val_acc_dual_encoder(graph_net_1, graph_net_2, val_batch,
                                                               fc, data_options, device)

                        val_count.append(validation_loss.detach())

                    except Exception as e:
                        print(f"Error {e}, batch: {val_batch}")
                        val_err_count += 1
                        traceback.print_exc()
                        continue

                validation_loss = (sum(val_count) / len(val_count)).detach()
                val_losses.append((validation_loss, j, i))

                print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print("Val acc: {}".format(validation_loss.detach()))

                print(f"Failed batches: {err_count}")

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

                graph_net_1.train()
                graph_net_2.train()

        if logging:
            wandb.log({"failed_batches": err_count})

    print(f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses


def val_acc_dual_encoder(model_1, model_2, batch, fc, data_options, device):

    data_1, data_2 = separate_link_batch(batch, data_options)

    graph_enc_1 = model_1(data_1.to(device))

    graph_enc_2 = model_2(data_2.to(device))

    preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds > 0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
