import logging
import pickle
import torch
import random
from utils.mongodb_utils import get_batches
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tqdm import tqdm
from torch_geometric.data.batch import Batch

from data.utils.graph_data_utils import DirectedData

from pymongo import MongoClient

client = MongoClient()
db = client['holist']
expr_collection = db['expression_graphs']


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
            return (ret['goal'], ret['thms'], ret['tac_id'], ret['thms_hard_negatives'])
        else:
            raise StopIteration


# todo standardise format with other data module
class HOListGraphModule(LightningDataModule):
    def __init__(self, dir, batch_size):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def load(self):
        self.vocab = torch.load(self.dir + 'vocab.pt')
        # self.expr_dict = torch.load(self.dir + 'expr_dict.pt')
        logging.info("Loading expression dictionary..")
        self.expr_dict = {k["_id"]: k['graph'] for k in tqdm(expr_collection.find({}))}
        self.train_data = torch.load(self.dir + 'train_data.pt')
        self.val_data = torch.load(self.dir + 'val_data.pt')
        self.thm_ls = torch.load(self.dir + 'train_thm_ls.pt')

    def setup(self, stage: str) -> None:
        source = self.config.source
        if source == 'mongodb':
            db = MongoClient()
            db = db[self.config.data_options['db']]
            expr_col = db[self.config.data_options['expression_col']]
            vocab_col = db[self.config.data_options['vocab_col']]
            split_col = db[self.config.data_options['split_col']]
            thms_col = db[self.config.data_options['thms_col']]

            self.vocab = {v["_id"]: v["index"]
                          for v in tqdm(vocab_col.find({}))
                          }

            self.thms_ls = [v['_id'] for v in thms_col.find({})]


            # load all examples in memory, otherwise keep as a cursor
            if self.config.data_options['dict_in_memory']:
                self.expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                                  for v in tqdm(expr_col.find({}))}
            else:
                self.expr_col = expr_col

            if self.config.data_options['split_in_memory']:
                # todo parametrise what fields are returned
                self.train_data = [(v["goal"], v["thms"], v['tac_id'], v['thms_hard_negatives'])
                             for v in tqdm(split_col.find({'train'}))]

                self.val_data = [(v["goal"], v["thms"], v['tac_id'], v['thms_hard_negatives'])
                                   for v in tqdm(split_col.find({'train'}))]
            else:
                self.train_data = MongoDataset(split_col.find({'split': 'train'}))
                self.val_data = MongoDataset(split_col.find({'split': 'val'}))


        elif source == 'directory':
            data_dir = self.config.data_options['directory']
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)

            self.vocab = self.data['vocab']
            self.expr_dict = self.data['expr_dict']
            self.expr_dict = {k: self.to_data(v) for k, v in self.expr_dict.items()}

            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
            self.thms_ls = self.data['train_thm_ls']

        else:
            raise NotImplementedError

    def setup(self, stage: str) -> None:
        if stage == "fit":
            logging.info("Loading data..")
            logging.info("Filtering data..")
            self.load()

            logging.info("Generating graph dictionary..")
            self.expr_dict = {k: DirectedData(
                x=torch.LongTensor(v['onehot']),
                # x=torch.LongTensor(
                #     [self.vocab[tok] if tok in self.vocab else self.vocab['UNK'] for tok in v['tokens']]),
                edge_index=torch.LongTensor(v['edge_index']),
                edge_attr=torch.LongTensor(v['edge_attr']),
                abs_pe=torch.LongTensor(v['depth']),
                attention_edge_index=torch.LongTensor(v['attention_edge_index']))
                for (k, v) in tqdm(self.expr_dict.items()) if 'attention_edge_index' in v}

            self.train_data = self.filter(self.train_data)
            self.val_data = self.filter(self.val_data)
            self.thms_ls = [d for d in self.thm_ls if d in self.expr_dict]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def gen_batch(self, batch):
        # todo filter negative sampling to be disjoint from positive samples

        # batch will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [self.expr_dict[x['goal']] for x in batch]

        # select random positive sample
        # if no parameters set it as a single element with '1' mapping to special token for no parameters
        pos_thms = [self.expr_dict[random.choice(x['thms'])] if len(x['thms']) > 0
                    else DirectedData(x=torch.LongTensor([1]), edge_index=torch.LongTensor([[], []]),
                                      edge_attr=torch.LongTensor([]), attention_edge_index=torch.LongTensor([[], []]),
                                      abs_pe=torch.LongTensor([0]))
                    for x in batch]

        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # 15 random negative samples per goal
        neg_thms = [[self.expr_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]

        goals = Batch.from_data_list(goals)
        pos_thms = Batch.from_data_list(pos_thms)
        neg_thms = [Batch.from_data_list(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    module = HOListGraphModule(dir='/home/sean/Documents/phd/deepmath-light/deepmath/processed_train_data/',
                               batch_size=16)
    module.setup("fit")
    print(next(iter(module.train_dataloader())))
    #
    loader = module.train_dataloader()
    i = 0
    for b in tqdm(loader):
        i += 1

    print(i)
    #
    #
