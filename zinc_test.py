from models.gnn_edge_labels import RegressionMLP, message_passing_gnn_edges
from torch import nn
import torch
from models.graph_transformers.SAT.sat.layers import AttentionRelations
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.data import Data
#%%
import lightning.pytorch as pl

class NewData(Data):
    def __init__(self, data):
        super().__init__()
        if data is None:
            return None
        self.softmax_idx = data.edge_index.size(1)
        self.x = data.x
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.y = data.y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'softmax_idx':
            return self.softmax_idx
        return super().__inc__(key, value, *args, **kwargs)

#%%
def transform(data):
    ret = NewData(data)
    return ret
#%%
# zd = ZINC(".", subset=True, transform=transform)
# loader = DataLoader(zd, batch_size=32)
# batch = next(iter(loader))
# batch.softmax_idx
#%%
def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

class ZincExperiment(pl.LightningModule):
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
        self.criterion = nn.L1Loss()

    def forward(self, graph):
        emb = self.embedding_model(graph)
        preds = self.classifier(emb)
        preds = torch.clip(preds, self.eps, 1 - self.eps)
        return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.criterion(preds, batch.y)
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.criterion(preds, batch.y)
        self.log("loss", loss, batch_size=self.batch_size, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

#%%
class ZincModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def setup(self, stage: str):
        if stage == "fit":
            self.train_pipe = ZINC(".", subset=True, transform=transform)
            self.val_pipe = ZINC(".", subset=True, transform=transform, split='val')
        if stage == "test":
            self.test_pipe = ZINC(".", subset=True, transform=transform, split='test')
    def train_dataloader(self):
        return DataLoader(self.train_pipe, batch_size=128)
    def val_dataloader(self):
        return DataLoader(self.val_pipe, batch_size=128)
    def test_dataloader(self):
        return DataLoader(self.test_pipe, batch_size=128)

    # only transfer relevant properties to device
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.x = batch.x.to(device).flatten()
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.ptr= batch.ptr.to(device)
        batch.batch = batch.batch.to(device)
        # batch.softmax_idx = batch.softmax_idx.tolist()
        batch.y = batch.y.to(device)
        return batch

torch.set_float32_matmul_precision('high')
#%%
module = ZincModule()
#%%
trainer = pl.Trainer(devices=1, accelerator='gpu')
#%%
# trainer.fit(ZincExperiment(AttentionRelations(28, 64), RegressionMLP(64)), module)
trainer.fit(ZincExperiment(message_passing_gnn_edges(28, 64, 4), RegressionMLP(64)), module)
