import itertools
import pickle
import random

import torch
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.utils.graph_data_utils import to_data, list_to_data
from experiments.pyrallis_configs import DataConfig
from utils.mongodb_utils import get_batches


class MongoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cursor, length, buf_size=4096):
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


'''
@todo:
holist integrated
    data module, mongo
    end to end

lean
    leangym wrapper
    leanstep mongo, loaded with vocab

Paper skeleton
    Intro
    
    Background
    
        Provers
            Lean, metamath (mizar), HOL4, HOL-Light 
        Current benchmarks
            Datasets
                LeanStep, mizar40, mizar60?, HOLStep, 
            Environments
                HOList, LeanGym, CoqGym, HOL4
        Approaches
            Task
                Supervised tasks, premise selection
            Proof Search
                BFS, Fringe, MCTS
            Model Architecture
                Embeddings
                    Transformer, LSTM, Tree-LSTM, GNN, BoW, ... (more detail in second part)
                Tactic/arg
                    GPT, fixed tactic MLPs, premise ranking
            Learning approach
                Pretrain, fine tune pipeline
                RL end to end
                
            AI-ITP component diagram of the above
        
        Embedding architectures
            Large background on GNN in ITP, Transformer more recently with Neural Theorem Proving and Lean(step/gym)
            GNN
            Transformer
            SAT
            Directed SAT
    
            
    Framework Overview
        Architecture diagram
        Case study/example?
        Metric? e.g. lines of code to add new setup  
        Vector database?
    
    Embedding Experiments
        Supervised
            HOList
            HOL4
            TacticZero
            HOLStep
            Mizar
            Lean?
            
        Ensemble?
        
        E2E
            HOList
            TacticZero
            Lean?
            
        Qualitative study
            Syntactic vs Semantic for TacticZero autoencoder vs fully trained
            Comparing embeddings between different systems, i.e. closest neighbors?
            
            
        
    
    
    
    
    
    
    
experiments
holist
holstep
tz vanilla + gnn + transformer + sat

tacticzero rl tidy 
tacticzero rl stats

Logging
Pyrallis configs for data generation 
datamodule stats?
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
                self.train_data = MongoDataset(split_col.find({'split': 'train'}),
                                               split_col.count_documents({'split': 'train'}))
                self.val_data = MongoDataset(split_col.find({'split': 'val'}),
                                             split_col.count_documents({'split': 'val'}))
                self.test_data = MongoDataset(split_col.find({'split': 'test'}),
                                              split_col.count_documents({'split': 'test'}))

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
