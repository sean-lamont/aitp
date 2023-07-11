import os
from tqdm import tqdm
from pymongo import MongoClient
from models import get_model
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch
import pickle
from lightning.pytorch import LightningDataModule


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
        # self.graph_dict = self.data['expr_dict']
        # self.vocab = self.data['vocab']

        self.graph_dict = {v["_id"]:
            v["graph"]
            for v in tqdm(col.find({}))
        }
        #
        # self.graph_dict = {v["_id"]:
        #     {
        #         'tokens': v["graph"]['tokens'],
        #         'edge_index': v["graph"]['edge_index'],
        #         'edge_attr': v["graph"]['edge_attr'],
        #         'attention_edge_index': v["graph"]['attention_edge_index'],
        #         'depth': v["graph"]['depth']}
        #
        #     for v in tqdm(col.find({}))
        # }
        #

        # if stage == "fit":
        #     self.train_data = self.data['mizar_labels'][:int(0.8 * len(self.data['mizar_labels']))]
        #     self.val_data = self.data['mizar_labels'][
        #                     int(0.8 * len(self.data['mizar_labels'])):int(0.9 * len(self.data['mizar_labels']))]
        # if stage == "test":
        #     self.test_data = self.data['mizar_labels'][int(0.9 * len(self.data['mizar_labels'])):]
        #
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
        return torch.utils.data.dataloader.DataLoader(self.train_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate)

    def val_dataloader(self):
        return torch.utils.data.dataloader.DataLoader(self.val_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate)

    def test_dataloader(self):
        return torch.utils.data.dataloader.DataLoader(self.test_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate())

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        data_1, data_2, y = batch

        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        y = y.to(device)

        return data_1, data_2, y

if __name__ == '__main__':
    module = MizarDataModule('/home/sean/Documents/phd/repo/aitp/data/mizar')
    module.setup('fit')
    # graph = next(iter(module.train_dataloader()))[0]

#
#     sat_config = {
#         "model_type": "sat",
#         # 'gnn_type': 'di_gcn',
#         "num_edge_features": 200,
#         "vocab_size": 13420,
#         "embedding_dim": 256,
#         "dim_feedforward": 256,
#         "num_heads": 2,
#         "num_layers": 2,
#         "in_embed": True,
#         "se": "formula-net",
#         # "se": "pna",
#         "abs_pe": False,
#         "abs_pe_dim": 128,
#         "use_edge_attr": True,
#         "dropout": 0.,
#         "gnn_layers": 3,
#         'small_inner': True,
#
# }
#
#
#     model = get_model.get_model(sat_config)
#     model(graph)
#
