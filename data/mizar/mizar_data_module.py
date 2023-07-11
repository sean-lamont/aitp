import pickle
from torch.utils.data.dataloader import DataLoader
import torch
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch_geometric.data import Batch
from torch_geometric.data import Data
from tqdm import tqdm


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


class DirectedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'attention_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


'''

Data Module for Graph based Models (Currently GNNs and Structure Aware Attention)

'''


class MizarDataModule(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        with open(self.dir + '/mizar_data_.pk', 'rb') as f:
            self.data = pickle.load(f)

        db = MongoClient()
        db = db['mizar']
        col = db['expression_graphs']

        with open(self.dir + '/mizar_data.pk', 'rb') as f:
            self.vocab = pickle.load(f)['vocab']

        self.graph_dict = {v["_id"]:
            v["graph"]
            for v in tqdm(col.find({}))
        }

        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def attention_collate(self, batch):
        y = torch.LongTensor([b[2] for b in batch])
        data_1 = [b[0] for b in batch]
        data_2 = [b[1] for b in batch]

        data_1 = Batch.from_data_list(
            [DirectedData(x=torch.LongTensor([self.vocab[a] for a in self.graph_dict[d]['tokens']]),
                          edge_index=torch.LongTensor(self.graph_dict[d]['edge_index']),
                          edge_attr=torch.LongTensor(self.graph_dict[d]['edge_attr']),
             attention_edge_index=torch.LongTensor(
                 self.graph_dict[d]['attention_edge_index']),
             abs_pe=torch.LongTensor(self.graph_dict[d]['depth']))
             for d in data_1])

        data_2 = Batch.from_data_list(
            [DirectedData(x=torch.LongTensor([self.vocab[a] for a in self.graph_dict[d]['tokens']]),
                          edge_index=torch.LongTensor(self.graph_dict[d]['edge_index']),
                          edge_attr=torch.LongTensor(self.graph_dict[d]['edge_attr']),
             attention_edge_index=torch.LongTensor(
                 self.graph_dict[d]['attention_edge_index']),
             abs_pe=torch.LongTensor(self.graph_dict[d]['depth']))
             for d in data_2])

        # unmasked (full n^2) attention edge index for undirected SAT models
        # data_1.attention_edge_index = ptr_to_complete_edge_index(data_1.ptr)
        # data_2.attention_edge_index = ptr_to_complete_edge_index(data_2.ptr)

        return data_1, data_2, y

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate)

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        data_1, data_2, y = batch

        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        y = y.to(device)

        return data_1, data_2, y