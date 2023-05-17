from models.gnn_edge_labels import RegressionMLP, message_passing_gnn_edges
import os
import pandas as pd
from torchvision import transforms
from torch import nn
import torch
from models.graph_transformers.SAT.sat.layers import AttentionRelations
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from models.graph_transformers.SAT.experiments.utils import *
from torch_geometric.data import Data
#%%
import lightning.pytorch as pl
from ogb.graphproppred import PygGraphPropPredDataset

max_seq_len = 5

def transform(data):
    ret = NewData(data)
    return ret

dataset = PygGraphPropPredDataset(name = "ogbg-code2", root = "./ogbg-code2/")

seq_len_list = np.array([len(seq) for seq in dataset.data.y])

print('Target seqence less or equal to {} is {}%.'.format(
    max_seq_len,
    np.sum(seq_len_list <= 1000) / len(seq_len_list))
)

split_idx = dataset.get_idx_split()

num_vocab = 5000

# building vocabulary for sequence predition. Only use training data.
vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)

# augment_edge: add next-token edge as well as inverse edges. add edge attributes.
# encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.

dataset.transform = transforms.Compose([
    augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len), transform
])

nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))



class NewData(Data):
    def __init__(self, data):
        super().__init__()
        if data is None:
            return None
        self.softmax_idx = data.edge_index.size(1)
        self.x = data.x
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.y = data.y_arr
        self.node_depth = data.node_depth

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'softmax_idx':
            return self.softmax_idx
        return super().__inc__(key, value, *args, **kwargs)


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

class OGBExperiment(pl.LightningModule):
    def __init__(self,
                 embedding_model,
                 classifier,
                 batch_size=128,
                 lr=1e-3):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.max_seq_len = 5

    def forward(self, graph):
        emb = self.embedding_model(graph)

        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](emb))
            preds = torch.stack(pred_list, dim=0).permute(1,2,0)
            return preds

        # preds = torch.clip(preds, self.eps, 1 - self.eps)
        # return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.criterion(preds, batch.y)
        self.log("loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.criterion(preds, batch.y)
        self.log("loss", loss, batch_size=self.batch_size, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class OGBModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        # self.dataset = PygGraphPropPredDataset(name="ogbg-code2", root="./ogbg-code2/")
        self.dataset = dataset
        self.split_idx = self.dataset.get_idx_split()

    def setup(self, stage: str):
        if stage == "fit":
            filter_mask = np.array([self.dataset[i].num_nodes for i in self.split_idx['train']]) <= 1000
            self.train_pipe = dataset[self.split_idx["train"][filter_mask]]
            filter_mask = np.array([self.dataset[i].num_nodes for i in self.split_idx['valid']]) <= 1000
            self.val_pipe = dataset[self.split_idx["valid"][filter_mask]]
        if stage == "test":
            filter_mask = np.array([self.dataset[i].num_nodes for i in self.split_idx['test']]) <= 1000
            self.test_pipe = dataset[self.split_idx["test"][filter_mask]]

    def train_dataloader(self):
        return DataLoader(self.train_pipe, batch_size=32, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_pipe, batch_size=32, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_pipe, batch_size=32, shuffle=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.x = batch.x.to(device)#.flatten()
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device).long()
        batch.ptr= batch.ptr.to(device)
        batch.batch = batch.batch.to(device)
        batch.softmax_idx = batch.softmax_idx.tolist()
        batch.y = batch.y.to(device)
        batch.node_depth = batch.node_depth.to(device)
        return batch

torch.set_float32_matmul_precision('high')

module = OGBModule()
trainer = pl.Trainer()

dim_hidden = 256
num_class = len(vocab2idx)

node_encoder = ASTNodeEncoder(
    dim_hidden,
    num_nodetypes=len(nodetypes_mapping['type']),
    num_nodeattributes=len(nodeattributes_mapping['attr']),
    max_depth=20
)

classifier = nn.ModuleList()
for i in range(max_seq_len):
    classifier.append(nn.Linear(dim_hidden, num_class))


trainer.fit(OGBExperiment(AttentionRelations(node_encoder,dim_hidden,2),classifier), module)

