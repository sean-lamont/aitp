from torch_geometric.data import Data
import lightning.pytorch as pl
import itertools
import random
from torch.nn.utils.rnn import pad_sequence
from utils.mongodb_utils import get_batches, get_all_batches
from tqdm import tqdm
from models.graph_transformers.SAT.sat.layers import AttentionRelations
import traceback
import wandb
from pymongo import MongoClient
from models import gnn_edge_labels, inner_embedding_network
from models.transformer_encoder_model import TransformerEmbedding
from models.graph_transformers.SAT.sat.models import GraphTransformer, AMRTransformer
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import DataLoader as StandardLoader
from torch.utils.data import TensorDataset


# todo Better way for softmax_idx, combined graph setup, positonal encoding, model saving/versioning, parametrised classifier, DAGLSTM?
class MaskData(Data):
    def __init__(self,x,edge_index=None,edge_attr=None,softmax_idx=None,mask_idx=None,mask_raw=None):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.sofmax_idx = softmax_idx
        self.mask_idx = mask_idx
        self.mask_raw = mask_raw

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'softmax_idx':
            return self.softmax_idx
        elif key == 'mask_idx':
            return self.x.size(0)

        return super().__inc__(key, value, *args, **kwargs)


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


# redundant
def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index

def to_batch_graph(batch, graph_collection, options):
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

    g,p,y = separate_link_batch(batch_, options)

    return g,p,y


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
        # data_1.softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_t])
        # data_2.softmax_idx = torch.cat([torch.tensor([0]), batch.softmax_idx_s])

        data_1.softmax_idx =  batch.softmax_idx_t
        data_2.softmax_idx =  batch.softmax_idx_s

    if 'edge_attr' in options:
        data_1.edge_attr = batch.edge_attr_t.long()
        data_2.edge_attr = batch.edge_attr_s.long()

    if 'attention_edge_index' in options:
        data_1.attention_edge_index = batch.attention_edge_index_t
        data_2.attention_edge_index = batch.attention_edge_index_s

    return data_1, data_2, torch.LongTensor(batch.y)



#todo leave padding and CLS etc for model
def to_batch_transformer(batch, graph_collection, options):

    stmts = list(set([sample['stmt'] for sample in batch]))
    conjs = list(set([sample['conj'] for sample in batch]))

    stmts.extend(conjs)

    exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))

    expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}


    # just use CLS token as separate (add 4 to everything)
    word_dict = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}


    #start of sentence is CLS
    conj_X = [torch.LongTensor([-3] + expr_dict[sample['conj']]['onehot']) + 4 for sample in batch]
    stmt_X = [torch.LongTensor([-3] + expr_dict[sample['stmt']]['onehot']) + 4 for sample in batch]

    Y = torch.LongTensor([sample['y'] for sample in batch])

    # batch_list = pad_sequence(conj_X), pad_sequence(stmt_X), Y

    # loader = iter(StandardLoader(batch_list), batch_size = len(batch))

    # batch_ = next(iter(loader))


    return pad_sequence(conj_X), pad_sequence(stmt_X), Y






class PremiseSelection(pl.LightningModule):
    def __init__(self, embedding_model_goal, embedding_model_premise, classifier, lr=1e-4):
        super().__init__()
        self.embedding_model_goal = embedding_model_goal
        self.embedding_model_premise = embedding_model_premise
        self.classifier = classifier
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = 32

    def forward(self, goal, premise):
        embedding_goal = self.embedding_model_goal(goal)

        embedding_premise = self.embedding_model_premise(premise)

        preds = self.classifier(torch.cat([embedding_goal, embedding_premise], dim = 1))

        preds = torch.clip(preds, self.eps, 1 - self.eps)

        return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        goal, premise, y = batch

        preds = self(goal, premise)

        loss = torch.nn.functional.cross_entropy(preds, y.float())

        self.log("loss", loss, batch_size=self.batch_size, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        goal, premise, y = batch

        preds =  self(goal, premise)

        preds = (preds > 0.5)

        acc = torch.sum(preds == y) / y.size(0)

        self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer


class PremiseSelectionExperiment:
    def __init__(self, config):
        self.model_config = config['model_config']
        self.data_config = config['data_config']
        self.exp_config = config['exp_config']

        self.data_options = self.data_config['data_options']
        self.source_config = self.data_config['source_config']

        self.embedding_dim = self.model_config['embedding_dim']

        self.lr = self.exp_config['learning_rate']
        self.weight_decay = self.exp_config['weight_decay']
        self.epochs = self.exp_config['epochs']
        self.batch_size = self.exp_config['batch_size']
        self.save = self.exp_config['model_save']
        self.val_size = self.exp_config['val_size']
        self.logging = self.exp_config['logging']
        self.device = self.exp_config['device']
        self.max_errors = self.exp_config['max_errors']
        self.val_frequency = self.exp_config['val_frequency']

        if self.data_config['data_type'] == 'standard_sequence':
            self.get_batch = to_batch_transformer

        elif self.data_config['data_type'] == 'graph':
            self.get_batch = to_batch_graph

        elif self.data_config['data_type'] == 'mask':
            self.get_batch = to_masked_batch_graph

    def get_model(self):
        if self.model_config['model_type'] == 'sat':
            return GraphTransformer(in_size=self.model_config['vocab_size'],
                                    num_class=2,
                                    d_model=self.model_config['embedding_dim'],
                                    dim_feedforward=self.model_config['dim_feedforward'],
                                    num_heads=self.model_config['num_heads'],
                                    num_layers=self.model_config['num_layers'],
                                    in_embed=self.model_config['in_embed'],
                                    se=self.model_config['se'],
                                    abs_pe=self.model_config['abs_pe'],
                                    abs_pe_dim=self.model_config['abs_pe_dim'],
                                    use_edge_attr=self.model_config['use_edge_attr'],
                                    num_edge_features=200,
                                    dropout=self.model_config['dropout'],
                                    k_hop=self.model_config['gnn_layers'])

        if self.model_config['model_type'] == 'amr':
            return AMRTransformer(in_size=self.model_config['vocab_size'],
                                  d_model=self.model_config['embedding_dim'],
                                  dim_feedforward=self.model_config['dim_feedforward'],
                                  num_heads=self.model_config['num_heads'],
                                  num_layers=self.model_config['num_layers'],
                                  in_embed=self.model_config['in_embed'],
                                  abs_pe=self.model_config['abs_pe'],
                                  abs_pe_dim=self.model_config['abs_pe_dim'],
                                  use_edge_attr=self.model_config['use_edge_attr'],
                                  num_edge_features=200,
                                  dropout=self.model_config['dropout'],
                                  layer_norm=True,
                                  global_pool='cls',
                                  device=self.model_config['device']
                                  )

        elif self.model_config['model_type'] == 'formula-net':
            return inner_embedding_network.FormulaNet(self.model_config['vocab_size'], self.model_config['embedding_dim'], self.model_config['gnn_layers'])

        elif self.model_config['model_type'] == 'formula-net-edges':
            return gnn_edge_labels.message_passing_gnn_edges(self.model_config['vocab_size'], self.model_config['embedding_dim'], self.model_config['gnn_layers'])

        elif self.model_config['model_type'] == 'digae':
            return None

        elif self.model_config['model_type'] == 'classifier':
            return None

        elif self.model_config['model_type'] == 'transformer':
            return TransformerEmbedding(ntoken=self.model_config['vocab_size'],
                                        d_model=self.model_config['embedding_dim'],
                                        nhead=self.model_config['num_heads'],
                                        nlayers=self.model_config['num_layers'],
                                        dropout=self.model_config['dropout'],
                                        d_hid=self.model_config['dim_feedforward'])

        elif self.model_config['model_type'] == 'transformer_relation':
            return AttentionRelations(ntoken = self.model_config['vocab_size'],
                                      # global_pool=False,
                                      embed_dim=self.model_config['embedding_dim'])
                                    #todo the rest

        else:
            return None




def test_iter(batches, batch_fn, graph_collection, options):
    for batch in batches:
        yield batch_fn(batch, graph_collection, options)



def val_iter(batches, batch_fn, graph_collection, options):
    for batch in itertools.cycle(batches):
        yield batch_fn(batch, graph_collection, options)


'''
Premise selection experiment with separate encoders for goal and premise
'''
class SeparateEncoderPremiseSelection(PremiseSelectionExperiment):
    def __int__(self, config):
        super(self, config)

    def run_lightning(self):

        torch.set_float32_matmul_precision('high')

        ps = PremiseSelection(self.get_model(),self.get_model(), gnn_edge_labels.F_c_module_(self.embedding_dim * 2))

        graph_collection, split_collection = get_data(self.source_config)

        train_cursor = split_collection.aggregate([{"$match": {"split": "train"}}, {"$sample": {"size": 10000000}}])
        val_cursor = split_collection.aggregate([{"$match": {"split": "valid"}}, {"$sample": {"size": self.val_size}}])

        train_batches = get_batches(train_cursor, self.batch_size)
        val_batches = get_all_batches(val_cursor, self.batch_size)

        # todo DataModule? also closing cursors cleaning up etc.
        train_loader = test_iter(train_batches, self.get_batch, graph_collection, self.data_options)
        val_loader = val_iter(val_batches, self.get_batch, graph_collection, self.data_options)

        # todo logging/wandb integration, model checkpoints

        trainer = pl.Trainer(val_check_interval=1000,
                             limit_val_batches=self.val_size // self.batch_size,
                             max_steps=1000,
                             enable_progress_bar=True,
                             log_every_n_steps=500,
                             # profiler='pytorch',
                             enable_checkpointing=False)



        # ps = torch.compile(ps)

        trainer.fit(model=ps, train_dataloaders=train_loader, val_dataloaders=val_loader)





# assume passed in set of expressions
def to_masked_batch_graph(graphs, options):
    batch_list = []

    for graph in graphs:
        graph = graph['graph']
        x =  graph['onehot']

        # todo mask edges as well
        # mask some tokens

        masking_rate = 0.15

        x = torch.LongTensor(x)

        #add 1 for mask
        x = x + 1

        mask = torch.rand(x.size(0)).ge(masking_rate)

        y = torch.index_select(x, (x * mask == 0).nonzero().flatten())

        x = x * mask
        mask_inds = (x == 0).nonzero().flatten()

        #need to set y as the values

        tmp_batch = MaskData(x = x)

        tmp_batch.y = y

        if 'edge_index' in options:
            if 'edge_index' in graph:
                tmp_batch.edge_index = torch.LongTensor(graph['edge_index'])
            else:
                raise NotImplementedError

        if 'edge_attr' in options:
            if 'edge_attr' in graph:
                tmp_batch.edge_attr = torch.LongTensor(graph['edge_attr'])
            else:
                raise NotImplementedError

        if 'attention_edge_index' in options:
            if 'attention_edge_index' in graph:
                tmp_batch.attention_edge_index = graph['attention_edge_index']
            else:
                # Default is global attention
                tmp_batch.attention_edge_index = torch.cartesian_prod(torch.arange(x.size(0)),
                                                                        torch.arange(x.size(0))).transpose(0,1)


        #todo make data options have possible values i.e. options['softmax_idx'] == AMR, use edges, else directed attention etc.
        if 'softmax_idx' in options:
            tmp_batch.softmax_idx = len(graph['edge_index'][0])

        tmp_batch.mask_idx = mask_inds
        tmp_batch.mask_raw = mask_inds
        # todo positional encoding including with depth

        batch_list.append(tmp_batch)


    loader = iter(DataLoader(batch_list, batch_size=len(batch_list)))

    batch_ = next(iter(loader))
    print (batch_.softmax_idx.size(0))

    return batch_




# todo make this with lightning


# criterion = torch.nn.CrossEntropyLoss()
# class MaskPretrain(PremiseSelectionExperiment):
#     def __int__(self, config):
#         super(self, config)
#
#     def run_mask_experiment(self):
#         self.graph_net = self.get_model().to(self.device)
#         print("Model details:")
#
#         print(self.graph_net)
#
#         if self.logging:
#             wandb.log({"Num_model_params": sum([p.numel() for p in self.graph_net.parameters() if p.requires_grad])})
#
#         fc = torch.nn.Sequential(torch.nn.Linear(self.model_config['embedding_dim'],self.model_config['embedding_dim']),
#                                             torch.nn.ReLU(),
#                                             torch.nn.LayerNorm(self.model_config['embedding_dim']),
#                                             torch.nn.Linear(self.model_config['embedding_dim'], self.model_config['vocab_size'])).to(self.device)
#
#         op_g = torch.optim.AdamW(self.graph_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         op_fc = torch.optim.AdamW(fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         training_losses = []
#
#         val_losses = []
#         best_acc = 0.
#
#         graph_collection, split_collection = get_data(self.source_config)
#
#         val_cursor = split_collection.find({"split":"valid"}).limit(self.val_size)
#
#         for j in range(self.epochs):
#             print(f"Epoch: {j}")
#             err_count = 0
#
#             # train_cursor = split_collection.aggregate([{"$match": {"split": "train"}}, {"$sample": {"size": 10000000}}])
#             train_cursor = graph_collection.find()
#             batches = get_batches(train_cursor, self.batch_size)
#
#             for i,db_batch in tqdm(enumerate(batches)):
#                 print (len(db_batch))
#                 try:
#                     # batch = to_batch_graph(db_batch, graph_collection, self.data_options)
#                     batch = self.get_batch(db_batch, self.data_options)
#                 except Exception as e:
#                     print(f"Error in batch: {e}")
#                     traceback.print_exc()
#                     continue
#
#                 # print (batch)
#                 op_g.zero_grad()
#                 op_fc.zero_grad()
#
#                 try:
#                     masked_encs = self.graph_net(batch.to(self.device))
#
#                     preds = fc(masked_encs)
#
#                     eps = 1e-6
#
#                     preds = torch.clip(preds, eps, 1 - eps)
#
#                     loss = criterion(torch.flatten(preds), batch.mask_idx.to(self.device))
#
#                     loss.backward()
#
#                     # op_enc.step()
#                     op_g.step()
#                     op_fc.step()
#
#
#                 except Exception as e:
#                     err_count += 1
#                     if err_count > self.max_errors:
#                         return Exception("Too many errors in training")
#                     print(f"Error in training {e}")
#                     traceback.print_exc()
#                     continue
#
#                 training_losses.append(loss.detach() / self.batch_size)
#
#                 if i % self.val_frequency == 0:
#                     print (sum(training_losses[-100:]) / len(training_losses[-100:]))
#



# todo add logic here for loading larger batch into memory, create dataloader from this then yield the next value in loader until done
# todo this should reduce the required number of queries
# def test_iter(batches, batch_fn, graph_collection, options):
#
#     for batch in batches:
#         # print (batch[0])
#
#         # hack it as list for now
#         batch = list(batch)
#         # print (batch[0])
#
#         stmts = list(set([sample['stmt'] for sample in batch]))
#         conjs = list(set([sample['conj'] for sample in batch]))
#
#         stmts.extend(conjs)
#
#         exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))
#
#         expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}
#
#         # print (len(expr_dict))
#         # print (list(expr_dict.keys())[0])
#
#
#         for i in range(0, len(batch), 32):
#             if (i + 1) * 32 >= len(batch):
#                 # print (f"fuck you cunt {i}")
#                 break
#             # print (batch_fn(batch[i * 32: (i + 1) * 32], graph_collection, options, expr_dict))
#             # exit()
#             yield batch_fn(batch[i * 32: (i + 1) * 32], graph_collection, options, expr_dict)

