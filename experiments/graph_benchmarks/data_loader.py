import os
from torch.utils.data import Dataset
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from ogb.graphproppred import Evaluator
import pickle
import torch
import pandas as pd
from torchvision import transforms
from experiments.graph_benchmarks.sat.utils import *
from torch import nn
from models.relation_transformer.relation_transformer import AttentionRelations
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import lightning.pytorch as pl
from ogb.graphproppred import PygGraphPropPredDataset
from models.gnn.formula_net.formula_net import FormulaNet

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # get sequence lengths
    # lengths = torch.tensor([ t.shape[0] for t in batch ])
    ## padd
    pad = lambda x: torch.nn.utils.rnn.pad_sequence(x)

    xi = pad([t[0] for t in batch])
    edge_attr = pad([t[1] for t in batch])
    xj = pad([t[2] for t in batch])
    y = batch[3]


    ## compute mask
    mask = (xi != 0)

    return xi, edge_attr, xj, y, mask

module.setup("fit")
module.setup("test")

train_ret = []
train_tensor = []
i = 0

for batch in tqdm(iter(module.train_dataloader())):
    i += 1

    if i % 33 == 0:
        # xi, edge_attr, xj, y, mask = collate_fn_padd(train_tensor)
        train_ret.append(collate_fn_padd(train_tensor))
        train_tensor = []

    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    node_depth = batch.node_depth
    y = batch.y
    x = torch.cat([x, node_depth], dim = 1)

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    train_tensor.append((xi, edge_attr, xj, y))

with open("/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/train_pad.pk", "wb") as f:
    pickle.dump(train_ret, f)

train_ret = []

val_ret = []
val_tensor = []
i = 0

for batch in tqdm(iter(module.val_dataloader())):
    i += 1

    if i % 33 == 0:
        # xi, edge_attr, xj, y, mask = collate_fn_padd(val_tensor)
        val_ret.append(collate_fn_padd(val_tensor))
        val_tensor = []

    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    node_depth = batch.node_depth
    y = batch.y
    x = torch.cat([x, node_depth], dim = 1)

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    val_tensor.append((xi, edge_attr, xj, y))

with open("/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/val_pad.pk", "wb") as f:
    pickle.dump(val_ret, f)


test_ret = []
test_tensor = []
i = 0

for batch in tqdm(iter(module.test_dataloader())):
    i += 1

    if i % 33 == 0:
        # xi, edge_attr, xj, y, mask = collate_fn_padd(test_tensor)
        test_ret.append(collate_fn_padd(test_tensor))
        test_tensor = []

    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    node_depth = batch.node_depth
    y = batch.y
    x = torch.cat([x, node_depth], dim = 1)

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    test_tensor.append((xi, edge_attr, xj, y))

with open("/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/test_pad.pk", "wb") as f:
    pickle.dump(test_ret, f)



class DataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

    def setup(self, stage: str):
        if stage == "fit":
            with open(self.train_file, "rb") as f:
                self.train_data = pickle.load(f)
            with open(self.val_file, "rb") as f:
                self.val_data = pickle.load(f)
        if stage == "test":
            with open(self.test_file, "rb") as f:
                self.test_data = pickle.load(f)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=1, shuffle=True, collate_fn= lambda x: x)#,num_workers=4)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=1, shuffle=False, collate_fn= lambda x: x)#,num_workers=4)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=False, collate_fn= lambda x: x)#,num_workers=4)


module = DataModule(train_file="/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/train_pad.pk",
                    val_file="/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/val_pad.pk",
                    test_file="/home/sean/Documents/phd/repo/aitp/experiments/graph_benchmarks/test_pad.pk")

module.setup("fit")
# module.setup("train")
loader = iter(module.train_dataloader())
for i in range(3):
    print (i)
    print (next(loader))
# print (batch[0].shape)

from models.relation_transformer.relation_transformer_new import AttentionRelations as RelationAttention

# model = RelationAttention(node_encoder, 256, 2)
# trainer.fit(OGBExperiment(node_encoder,model,classifier), module)


# xi, edge_attr, xj, y, mask = collate_fn_padd(val_tensor)
# print (xi.shape, edge_attr.shape, xj.shape, y.shape, mask.shape)
#

