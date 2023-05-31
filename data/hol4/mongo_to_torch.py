import torch_geometric.loader.dataloader
from pymongo import MongoClient
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from collections import namedtuple
import torch
import os
from lightning.pytorch import LightningDataModule


class HOL4DataModule(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            expr = db.expression_graph_data
            meta = db.expression_metadata

            graph_dict = {v['_id']: v['graph'] for v in expr.find({})}
            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression']) for v in
                         meta.find({})}

            train_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in
                          split.find({}) if
                          v['split'] == 'train']
            val_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in split.find({})
                        if
                        v['split'] == 'val']
            test_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in
                         split.find({}) if
                         v['split'] == 'test']

            data = {'graph_dict': graph_dict, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}
            torch.save(data, self.dir + "/data.pt")

    def to_relation(self, data):
        data_1, data_2, y = data

        x = data_1['onehot']
        edge_index = data_1['edge_index']
        edge_attr = torch.LongTensor(data_1['edge_attr']) + 1
        xi = torch.LongTensor([x[i] for i in edge_index[0]]) + 1
        xj = torch.LongTensor([x[i] for i in edge_index[1]]) + 1
        data_1 = Data(xi=xi, xj=xj, edge_attr_=edge_attr)

        x = data_2['onehot']
        edge_index = data_2['edge_index']
        edge_attr = torch.LongTensor(data_2['edge_attr']) + 1
        xi = torch.LongTensor([x[i] for i in edge_index[0]]) + 1
        xj = torch.LongTensor([x[i] for i in edge_index[1]]) + 1
        data_2 = Data(xi=xi, xj=xj, edge_attr_=edge_attr)

        return data_1, data_2, y

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def collate_pad(self, batch):
        xis1 = []
        xis2 = []
        xjs1 = []
        xjs2 = []
        edge_attrs1 = []
        edge_attrs2 = []
        ys = []

        for data_1, data_2, y in batch:
            xis1.append(data_1.xi)
            xjs1.append(data_1.xj)
            edge_attrs1.append(data_1.edge_attr_)
            xis2.append(data_2.xi)
            xjs2.append(data_2.xj)
            edge_attrs2.append(data_2.edge_attr_)
            ys.append(y)

        xi1 = torch.nn.utils.rnn.pad_sequence(xis1)
        xj1 = torch.nn.utils.rnn.pad_sequence(xjs1)
        edge_attr_1 = torch.nn.utils.rnn.pad_sequence(edge_attrs1)

        mask1 = (xi1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        xi2 = torch.nn.utils.rnn.pad_sequence(xis2)
        xj2 = torch.nn.utils.rnn.pad_sequence(xjs2)
        edge_attr_2 = torch.nn.utils.rnn.pad_sequence(edge_attrs2)

        mask2 = (xi2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return Data(xi=xi1, xj=xj1, edge_attr_=edge_attr_1, mask=mask1), Data(xi=xi2, xj=xj2, edge_attr_=edge_attr_2,
                                                                              mask=mask2), torch.LongTensor(ys)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_pad)


class HOL4DataModuleGraph(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            expr = db.expression_graph_data
            meta = db.expression_metadata

            graph_dict = {v['_id']: v['graph'] for v in expr.find({})}
            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression']) for v in
                         meta.find({})}

            train_data = [self.to_graph((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in split.find({})
                          if
                          v['split'] == 'train']
            val_data = [self.to_graph((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in split.find({}) if
                        v['split'] == 'val']
            test_data = [self.to_graph((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in split.find({})
                         if
                         v['split'] == 'test']

            data = {'graph_dict': graph_dict, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}
            torch.save(data, self.dir + "/data.pt")

    def to_graph(self, data):
        data_1, data_2, y = data

        x = torch.LongTensor(data_1['onehot'])
        edge_index = torch.LongTensor(data_1['edge_index'])
        edge_attr = torch.LongTensor(data_1['edge_attr'])
        data_1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = torch.LongTensor(data_2['onehot'])
        edge_index = torch.LongTensor(data_2['edge_index'])
        edge_attr = torch.LongTensor(data_2['edge_attr'])
        data_2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data_1, data_2, y

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def train_dataloader(self):
        return torch_geometric.loader.dataloader.DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch_geometric.loader.dataloader.DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch_geometric.loader.dataloader.DataLoader(self.test_data, batch_size=self.batch_size)


# Sequence data class with all tokens (including e.g @ symbols) and keeping position
class HOL4SequenceModule(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            meta = db.expression_metadata

            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression'])
                         for v in meta.find({})}

            # build vocab
            vocab = {}
            i = 1
            for k in expr_dict.keys():
                tmp = k.split(" ")
                for t in tmp:
                    if t not in vocab:
                        vocab[t] = i
                        i += 1

            train_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                          v['split'] == 'train']
            val_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                        v['split'] == 'valid']
            test_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                         v['split'] == 'test']

            data = {'vocab': vocab, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}
            torch.save(data, self.dir + "/data.pt")

    def to_sequence(self, data, vocab):
        data_1, data_2, y = data

        data_1 = data_1.split(" ")
        data_1 = torch.LongTensor([vocab[c] for c in data_1])

        data_2 = data_2.split(" ")
        data_2 = torch.LongTensor([vocab[c] for c in data_2])

        return data_1, data_2, y

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def collate_pad(self, batch):
        x1 = []
        x2 = []
        ys = []
        for data_1, data_2, y in batch:
            x1.append(data_1)
            x2.append(data_2)
            ys.append(y)

        x1 = torch.nn.utils.rnn.pad_sequence(x1)
        x1 = x1[:300]
        mask1 = (x1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        x2 = torch.nn.utils.rnn.pad_sequence(x2)
        x2 = x2[:300]
        mask2 = (x2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return (x1, mask1), (x2, mask2), torch.LongTensor(ys)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_pad)
