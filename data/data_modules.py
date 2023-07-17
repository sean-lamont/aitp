import pickle
import random
from utils.mongodb_utils import get_batches
from data.utils.graph_data_utils import ptr_to_complete_edge_index, DirectedData, list_to_graph, list_to_sequence, \
    list_to_relation, to_data, list_to_data
import pyrallis
from experiments.pyrallis_configs import DataConfig
from abc import abstractmethod
from torch.utils.data.dataloader import DataLoader
import torch
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch_geometric.data import Batch
from torch_geometric.data import Data
from tqdm import tqdm


class MongoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cursor, buf_size=4096):
        super(MongoDataset).__init__()
        self.cursor = cursor
        self.batches = get_batches(self.cursor, batch_size=buf_size)
        self.curr_batches = next(self.batches)
        self.remaining = len(self.curr_batches)

    def __iter__(self):
        return self

    # todo figure out why it stops after one iteration
    def __next__(self):
        if self.remaining == 0:
            self.curr_batches = next(self.batches)
            random.shuffle(self.curr_batches)
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            ret = self.curr_batches.pop()
            # todo parametrise what fields are returned
            return (ret['conj'], ret['stmt'], ret['y'])
        else:
            raise StopIteration


'''
@todo:
abstract 
tpr

holist integrated

Pyrallis configs for data generation (HOL4, HOList, HOLStep)
experiments
holist
holstep
add holist to aitp
holist data module inheritence
tacticzero vanilla

tacticzero rl tidy 
tacticzero rl log probs and stats

logging throughout
datamodule stats
leanstep + leangym
Dataset/H5? 

'''

'''
Data Module takes in:

    - config, with source, either directory with a data dictionary or a MongoDB collection
        - if {'source': 'dir', 'attributes': {'dir': directory}} 
                or {'source': mongodb, 'attributes':{'database':.., 'collection':..} 
    - batch_size 
    - attention_edge, pe for graph, max_seq_len for sequence

'''


class PremiseDataModule(LightningDataModule):
    def __init__(self, config: DataConfig):

        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        source = self.config.source
        if source == 'mongodb':
            db = MongoClient()
            db = db[self.config.data_options['db']]
            expr_col = db[self.config.data_options['expression_col']]
            vocab_col = db[self.config.data_options['vocab_col']]
            split_col = db[self.config.data_options['split_col']]

            self.vocab = {v["_id"]: v["index"]
                          for v in tqdm(vocab_col.find({}))
                          }

            # if dict_in_memory save all expressions to disk, otherwise return cursor
            if self.config.data_options['dict_in_memory']:
                self.expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                                  for v in tqdm(expr_col.find({}))}
            else:
                self.expr_col = expr_col

            # if data_in_memory, save all examples to disk
            if self.config.data_options['split_in_memory']:
                # todo paramatrise what fields are returned
                self.data = [(v["conj"], v["stmt"], v['y'], v['split'])
                             for v in tqdm(split_col.find({}))]
                self.train_data = [d for d in self.data if d[3] == 'train']
                self.val_data = [d for d in self.data if d[3] == 'val']
                self.test_data = [d for d in self.data if d[3] == 'test']
            else:
                self.train_data = MongoDataset(split_col.find({'split': 'train'}))
                self.val_data = MongoDataset(split_col.find({'split': 'val'}))
                self.test_data = MongoDataset(split_col.find({'split': 'test'}))

        elif source == 'directory':
            data_dir = self.config.data_options['directory']
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)

            self.vocab = self.data['vocab']
            self.expr_dict = self.data['expr_dict']
            self.expr_dict = {k: self.to_data(v) for k, v in self.expr_dict.items()}

            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
            self.test_data = self.data['test_data']

        else:
            raise NotImplementedError

    def list_to_data(self, data_list):
        # either stream from database when too large for memory, or we can save all to disk for quick access
        if self.config.source == 'mongodb' and not self.config.data_options['dict_in_memory']:
            tmp_expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                             for v in self.expr_col.find({'_id': {'$in': data_list}})}
            batch = [tmp_expr_dict[d] for d in data_list]
        else:
            batch = [self.expr_dict[d] for d in data_list]

        return list_to_data(batch, config=self.config)


    def to_data(self, expr):
        return to_data(expr, self.config.type, self.vocab, self.config)

    def collate_data(self, batch):
        y = torch.LongTensor([b[2] for b in batch])
        data_1 = self.list_to_data([b[0] for b in batch])
        data_2 = self.list_to_data([b[1] for b in batch])
        return data_1, data_2, y

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=self.config.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=self.config.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data)

#
# '''
#
# Data Module for Graph based Models (Currently GNNs and Structure Aware Attention)
#
# '''
#
#
# class GraphDataModule(PremiseDataModule):
#     def __init__(self, config: DataConfig):
#         super().__init__(config)
#         self.config = config
#
#     def setup(self, stage: str) -> None:
#         super(GraphDataModule, self).setup(stage)
#         # add e.g. thm_ls for holist here
#
#     def list_to_data(self, data_list):
#         data_list = super(GraphDataModule, self).list_to_data(data_list)
#         return list_to_graph(data_list, self.config.attributes)
#
# class SequenceDataModule(PremiseDataModule):
#     def __init__(self, config: DataConfig):
#         super().__init__(config)
#         self.config = config
#         self.max_len = self.config.attributes['max_len']
#
#     # Convert an expression to a LongTensor
#     def list_to_data(self, data_list):
#         data_list = super(SequenceDataModule, self).list_to_data(data_list)
#         return list_to_sequence(data_list, self.max_len)
#
#
# class RelationDataModule(PremiseDataModule):
#     def __init__(self, config: DataConfig):
#         super().__init__(config)
#         self.config = config
#         self.max_len = self.config.attributes['max_len']
#
#     # Take tokens from standard graph and generate (source, target, edge) relations
#     def list_to_data(self, data_list):
#         data_list = super(RelationDataModule, self).list_to_data(data_list)
#         return list_to_relation(data_list, self.max_len)