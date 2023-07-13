import pickle
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

'''
@todo:
Mongo + file setup scripts for everything
abstract 
tpr

add holist to aitp
holist data module inheritence
tacticzero rl with pyrallis 
tacticzero rl tidy 
lean 


experiments

    holist
    holstep
    tacticzero
    mizar?

'''


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
Data Module takes in:

    - config, with source, either directory with a data dictionary or a MongoDB collection
        - if {'source': 'dir', 'attributes': {'dir': directory}} or {'source': mongodb, 'attributes':{'database':.., 'collection':..} 
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
            expr_col = db[self.config.data_options['expressions']]
            vocab_col = db[self.config.data_options['vocab']]
            split_col = db[self.config.data_options['split_data']]

            self.vocab = {v["_id"]: v["index"]
                          for v in tqdm(vocab_col.find({}))
                          }

            # if dict_in_memory save all expressions to disk, otherwise return cursor
            if self.config.data_options['dict_in_memory']:
                self.expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.attributes['filter']})
                                  for v in tqdm(expr_col.find({}))}

            else:
                self.expr_col = expr_col

            # if data_in_memory, save all examples to disk
            if self.config.data_options['split_in_memory']:
                self.data = [(v["conj"], v["stmt"], v['y'], v['split'])
                             for v in tqdm(split_col.find({}))]
                self.train_data = [d for d in self.data if d[3] == 'train']
                self.val_data = [d for d in self.data if d[3] == 'val']
                self.test_data = [d for d in self.data if d[3] == 'test']
            else:
                self.train_data = split_col.find({'split': 'train'})
                self.val_data = split_col.find({'split': 'val'})
                self.test_data = split_col.find({'split': 'test'})

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
            tmp_expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.attributes['filter']})
                             for v in self.expr_col.find({'_id': {'$in': data_list}})}
            return [self.to_data(tmp_expr_dict[d]) for d in data_list]
        else:
            return [self.expr_dict[d] for d in data_list]

    @abstractmethod
    def to_data(self, expr):
        pass

    @abstractmethod
    def collate_data(self, batch):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data)


'''

Data Module for Graph based Models (Currently GNNs and Structure Aware Attention)

'''


class GraphDataModule(PremiseDataModule):
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.config = config

    def setup(self, stage: str) -> None:
        super(GraphDataModule, self).setup(stage)
        # add e.g. thm_ls for holist here

    '''

    Convert an expression dictionary to a PyG Data object, with optional Positional Encoding and Attention Masking

    '''

    def to_data(self, expr):
        if 'attention_edge_index' not in self.config.attributes or self.config.attributes['attention_edge'] == 'full':
            data = DirectedData(x=torch.LongTensor([self.vocab[a] for a in expr['tokens']]),
                                edge_index=torch.LongTensor(expr['edge_index']),
                                edge_attr=torch.LongTensor(expr['edge_attr']), )
        elif self.config.attributes['attention_edge'] == 'directed':
            data = DirectedData(x=torch.LongTensor([self.vocab[a] for a in expr['tokens']]),
                                edge_index=torch.LongTensor(expr['edge_index']),
                                edge_attr=torch.LongTensor(expr['edge_attr']),
                                attention_edge_index=torch.LongTensor(
                                    self.expr_dict[expr]['attention_edge_index']))
        else:
            raise NotImplementedError
        if 'pe' in self.config.attributes:
            data.abs_pe = self.expr_dict[expr][self.config.pe]

        return data

    def collate_data(self, batch):
        y = torch.LongTensor([b[2] for b in batch])
        data_1 = [b[0] for b in batch]
        data_2 = [b[1] for b in batch]

        data_1 = Batch.from_data_list(self.list_to_data(data_1))
        data_2 = Batch.from_data_list(self.list_to_data(data_2))

        if 'attention_edge' in self.config.attributes and self.config.attributes['attention_edge'] == 'full':
            data_1.attention_edge_index = ptr_to_complete_edge_index(data_1.ptr)
            data_2.attention_edge_index = ptr_to_complete_edge_index(data_2.ptr)

        return data_1, data_2, y


class SequenceDataModule(PremiseDataModule):
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.config = config
        self.max_len = self.config.attributes['max_len']

    '''

    Convert an expression to a LongTensor
    @todo: Add Positional Encoding here?

    '''

    def to_data(self, expr):
        return torch.LongTensor([self.vocab[a] for a in self.expr_dict[expr]['tokens']])

    def collate_data(self, batch):
        y = torch.LongTensor([b[2] for b in batch])
        data_1 = self.list_to_data([b[0] for b in batch])
        data_2 = self.list_to_data([b[1] for b in batch])

        data_1 = torch.nn.utils.rnn.pad_sequence(data_1)
        data_1 = data_1[:self.max_len]
        mask1 = (data_1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        data_2 = torch.nn.utils.rnn.pad_sequence(data_2)
        data_2 = data_2[:self.max_len]
        mask2 = (data_2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return (data_1, mask1), (data_2, mask2), y


class RelationDataModule(PremiseDataModule):
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.config = config
        self.max_len = self.config.attributes['max_len']

    '''
    
    Take tokens from standard graph and generate (source, target, edge) relations
    
    '''

    def to_data(self, expr):
        x = [self.vocab[a] for a in expr['tokens']]
        edge_index = expr['edge_index']
        edge_attr = torch.LongTensor(expr['edge_attr'])
        xi = torch.LongTensor([x[i] for i in edge_index[0]])
        xj = torch.LongTensor([x[i] for i in edge_index[1]])
        return (xi, xj, edge_attr)

    def collate_data(self, batch):
        data_1 = self.list_to_data([b[0] for b in batch])
        data_2 = self.list_to_data([b[1] for b in batch])
        y = torch.LongTensor([b[2] for b in batch])

        xis1 = [d[0] for d in data_1][:self.max_len]
        xjs1 = [d[1] for d in data_1][:self.max_len]
        edge_attrs1 = [d[2] for d in data_1][:self.max_len]

        xis2 = [d[0] for d in data_2][:self.max_len]
        xjs2 = [d[1] for d in data_2][:self.max_len]
        edge_attrs2 = [d[2] for d in data_2][:self.max_len]

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

        return (Data(xi=xi1, xj=xj1, edge_attr_=edge_attr_1, mask=mask1),
                Data(xi=xi2, xj=xj2, edge_attr_=edge_attr_2,
                     mask=mask2), y)


def main():
    cfg = pyrallis.parse(config_class=DataConfig)
    # cfg.source = 'directory'
    # cfg.data_options = {'directory': '/home/sean/Documents/phd/repo/aitp/data/mizar/mizar_data_new.pk'}
    cfg.source = 'mongodb'
    cfg.data_options = {'db': 'mizar', 'expressions': 'expression_graphs'}
    cfg.attributes = {'attention_edge': 'full', 'pe': False,
                      'filter': ['edge_index', 'tokens', 'edge_attr', 'attention_edge_index']}
    module = GraphDataModule(cfg)
    module.setup("fit")
    print(next(iter(module.train_dataloader())))


if __name__ == '__main__':
    main()
