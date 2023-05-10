import torchdata
import pickle
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from data.utils.hd5_utils import to_hdf5, to_pickle, PremiseSelectionDataset
import h5py
import lightning.pytorch as pl
from utils.mongodb_utils import get_batches, get_all_batches
from tqdm import tqdm
from pymongo import MongoClient
import torch


class LinkData(Data):
    def __init__(self, edge_index_s=None,
                 x_s=None, edge_index_t=None,
                 x_t=None, edge_attr_s=None,
                 edge_attr_t=None,
                 softmax_idx_s=None,
                 softmax_idx_t=None,
                 y=None):

        super().__init__()

        self.edge_index_s = edge_index_s
        self.x_s = x_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t

        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t

        # softmax index used for AMR model
        self.softmax_idx_s = softmax_idx_s
        self.softmax_idx_t = softmax_idx_t

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s' or key == 'attention_edge_index_s':
            return self.x_s.size(0)

        elif key == 'edge_index_t' or key == 'attention_edge_index_t':
            return self.x_t.size(0)

        elif key == 'softmax_idx_s':
            return self.softmax_idx_s

        elif key == 'softmax_idx_t':
            return self.softmax_idx_t

        return super().__inc__(key, value, *args, **kwargs)


class MongoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cursor, buf_size):
        super(MongoDataset).__init__()
        self.cursor = cursor
        self.batches = get_batches(self.cursor, batch_size=buf_size)
        self.curr_batches = next(self.batches)
        self.remaining = len(self.curr_batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            self.curr_batches = next(self.batches)
            self.remaining = len(self.curr_batches)

        self.remaining -= 1

        if self.remaining >= 0:
            return self.curr_batches.pop()
        else:
            raise StopIteration


class MongoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        client = MongoClient()
        self.db = client[self.config['db_name']]
        self.collection = self.db[self.config['collection_name']]
        self.batch_size = self.config['batch_size']
        self.options = config['options']

    def sample_to_link(self, sample):
        options = self.options
        stmt_graph = sample['stmt_graph']
        conj_graph = sample['conj_graph']
        y = sample['y']

        x1 = conj_graph['onehot']
        x1_mat = torch.LongTensor(x1)

        x2 = stmt_graph['onehot']
        x2_mat = torch.LongTensor(x2)

        ret = LinkData(x_s=x2_mat, x_t=x1_mat, y=torch.tensor(y))

        if 'edge_index' in options:
            if 'edge_index' in conj_graph and 'edge_index' in stmt_graph:
                x1_edge_index = conj_graph['edge_index']
                x1_edge_index = torch.LongTensor(x1_edge_index)

                x2_edge_index = stmt_graph['edge_index']
                x2_edge_index = torch.LongTensor(x2_edge_index)

                ret.edge_index_t = x1_edge_index
                ret.edge_index_s = x2_edge_index
            else:
                raise NotImplementedError

        if 'edge_attr' in options:
            if 'edge_attr' in conj_graph and 'edge_attr' in stmt_graph:
                x1_edge_attr = conj_graph['edge_attr']
                x1_edge_attr = torch.LongTensor(x1_edge_attr)

                x2_edge_attr = stmt_graph['edge_attr']
                x2_edge_attr = torch.LongTensor(x2_edge_attr)

                ret.edge_attr_t = x1_edge_attr
                ret.edge_attr_s = x2_edge_attr
            else:
                raise NotImplementedError

        # Edge index used to determine where attention is propagated in Message Passing Attention schemes

        if 'attention_edge_index' in options:
            if 'attention_edge_index' in conj_graph and 'attention_edge_index' in stmt_graph:
                ret.attention_edge_index_t = conj_graph['attention_edge_index']
                ret.attention_edge_index_s = stmt_graph['attention_edge_index']
            else:
                raise NotImplementedError


        if 'softmax_idx' in options:
            ret.softmax_idx_t = x1_edge_index.size(1)
            ret.softmax_idx_s = x2_edge_index.size(1)

        return ret

    def custom_collate(self, data):
        data_list = [self.sample_to_link(d) for d in data]
        batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_t'])
        return separate_link_batch(batch)


    def setup(self, stage: str):
        if stage == "fit":
            self.train_cursor = self.collection.find({"split": "train"}).sort("rand_idx", 1)
            self.train_data = MongoDataset(self.train_cursor, self.config['buf_size'])

            self.val_cursor = self.collection.find({"split": "valid"}).sort("rand_idx",1)
            self.val_data = MongoDataset(self.val_cursor, self.config['buf_size'])

        if stage == "test":
            self.test_cursor = self.collection.find({"split": "test"}).sort("rand_idx", 1)
            self.test_data = MongoDataset(self.test_cursor, self.config['buf_size'])

        # if stage == "predict":

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)


def separate_link_batch(batch):
    # assume data will always have at least x variable
    data_1 = Data(x=batch.x_t, batch=batch.x_t_batch, ptr=batch.x_t_ptr)
    data_2 = Data(x=batch.x_s, batch=batch.x_s_batch, ptr=batch.x_s_ptr)

    if hasattr(batch, 'edge_index_t'):
        data_1.edge_index = batch.edge_index_t
        data_2.edge_index = batch.edge_index_s

    if hasattr(batch,'softmax_idx_t'):
        data_1.softmax_idx =  batch.softmax_idx_t
        data_2.softmax_idx =  batch.softmax_idx_s

    if hasattr(batch, 'edge_attr_t'):
        data_1.edge_attr = batch.edge_attr_t.long()
        data_2.edge_attr = batch.edge_attr_s.long()

    if hasattr(batch, 'attention_edge_index_t'):
        data_1.attention_edge_index = batch.attention_edge_index_t
        data_2.attention_edge_index = batch.attention_edge_index_s

    return data_1, data_2, batch.y



data_dir = '/home/sean/Documents/phd/aitp/data/utils/processed_data'

# save preprocessed graph data in batches to files. Prevents need for recomputing batch level attributes,
# such as ptr, batch idx etc.
def write_mongo_to_h5():
    data_module = MongoDataModule(config={'buf_size': 2048, 'batch_size': 32, 'db_name': 'hol_step',
                                                         'collection_name': 'pretrain_graphs',
                                                         'options': ['edge_attr', 'edge_index', 'softmax_idx']})

    data_module.setup('fit')
    data_module.setup('test')

    train_loader = iter(data_module.train_dataloader())
    val_loader = iter(data_module.val_dataloader())
    test_loader = iter(data_module.test_dataloader())


    BATCHES_PER_FILE = 512

    def batch_to_h5(loader, name):
        data_list = []
        file_num = 0

        for i, batch in tqdm(enumerate(loader)):
            data_list.append(batch)

            if i > 0 and i % BATCHES_PER_FILE == 0:
                with h5py.File(data_dir + f'/{name}_{file_num}.h5', 'w') as f:
                    to_hdf5(data_list, f, 'data')
                data_list = []
                file_num += 1

    batch_to_h5(train_loader, 'train')
    batch_to_h5(val_loader, 'val')
    batch_to_h5(test_loader, 'test')


@torchdata.datapipes.functional_datapipe("load_h5_data")
class H5DataLoader(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        # self.transform = kwargs['transform']
    # def file_to_data(self, h5file):

    def __iter__(self):
        for file_name in self.source_datapipe:
            with h5py.File(file_name, 'r') as h5file:
                x1 = torch.from_numpy(h5file['data/x1'][:])
                edge_index1 = torch.from_numpy(h5file['data/edge_index1'][:])
                edge_attr1 = torch.from_numpy(h5file['data/edge_attr1'][:])
                batch1 = torch.from_numpy(h5file['data/batch1'][:])
                ptr1 = torch.from_numpy(h5file['data/ptr1'][:])
                edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:])#.tolist()
                # edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:]).tolist()

                x2 = torch.from_numpy(h5file['data/x2'][:])
                edge_index2 = torch.from_numpy(h5file['data/edge_index2'][:])
                edge_attr2 = torch.from_numpy(h5file['data/edge_attr2'][:])
                batch2 = torch.from_numpy(h5file['data/batch2'][:])
                ptr2 = torch.from_numpy(h5file['data/ptr2'][:])
                # edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:]).tolist()
                edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:])#.tolist()

                y = torch.from_numpy(h5file['data/y'][:])

                data_len_1 = torch.from_numpy(h5file['data/x1_len'][:])
                data_len_2 = torch.from_numpy(h5file['data/x2_len'][:])

                for i in range(len(x1)):
                    num_nodes1 = data_len_1[i][0]
                    num_edges1 = data_len_1[i][1]

                    num_nodes2 = data_len_2[i][0]
                    num_edges2 = data_len_2[i][1]


                    # print (f"ptr {ptr1[i]}")

                    # yield (x1[i, :num_nodes1],
                    #            edge_index1[i, :, :num_edges1],
                    #            edge_attr1[i, :num_edges1],
                    #            batch1[i, :num_nodes1],
                    #            ptr1[i],
                    #            edge_ptr1[i]), \
                    #         (x2[i, :num_nodes2],
                    #         edge_index2[i, :, :num_edges2],
                    #         edge_attr2[i, :num_edges2],
                    #         batch2[i, :num_nodes2],
                    #         ptr2[i],
                    #         edge_ptr2[i]), \
                    #         y[i]



                    # X1 = torch.tensor_split(x1[i, :num_nodes1],ptr1[i][1:-1].long())
                    # X2 = torch.tensor_split(x2[i, :num_nodes2],ptr2[i][1:-1].long())
                    #
                    # X1 = torch.nn.utils.rnn.pad_sequence(X1)
                    # X2 = torch.nn.utils.rnn.pad_sequence(X2)
                    #
                    # yield X1, X2, y[i]

                    data_1 = Data(x=x1[i, :num_nodes1],
                               edge_index=edge_index1[i, :, :num_edges1],
                               edge_attr=edge_attr1[i, :num_edges1],
                               batch=batch1[i, :num_nodes1],
                               ptr=ptr1[i],
                               softmax_idx=edge_ptr1[i])

                    data_1.pin_memory()

                    data_2 = Data(x=x2[i, :num_nodes2],
                             edge_index=edge_index2[i, :, :num_edges2],
                             edge_attr=edge_attr2[i, :num_edges2],
                             batch=batch2[i, :num_nodes2],
                             ptr=ptr2[i],
                             softmax_idx=edge_ptr2[i])

                    data_2.pin_memory()

                    y_i = y[i]

                    yield data_1, data_2, y_i


def build_h5_datapipe(masks) -> torchdata.datapipes.iter.IterDataPipe:
    datapipe = torchdata.datapipes.iter.FileLister(data_dir, masks=masks)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.load_pickle_data(transform=cfg.transform)
    datapipe = datapipe.load_h5_data()
    return datapipe


class H5DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # todo
        if False:
            write_mongo_to_h5()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_pipe = build_h5_datapipe("train*")
            self.val_pipe = build_h5_datapipe("val*")

        if stage == "test":
            self.test_pipe = build_h5_datapipe("test*")

    def train_dataloader(self):
         return torch.utils.data.DataLoader(
            dataset=self.train_pipe,
            batch_size=1,
            # shuffle=True,
            # drop_last=True,
            pin_memory=True,
            num_workers=0, collate_fn=lambda x: x)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_pipe,
            batch_size=1,
            # shuffle=True,
            # drop_last=True,
            num_workers=0, collate_fn=lambda x: x)

    def test_dataloader(self):

        return torch.utils.data.DataLoader(
            dataset=self.test_pipe,
            batch_size=1,
            # shuffle=True,
            # drop_last=True,
            num_workers=0, collate_fn=lambda x: x)









# 8m 35s, 800M
# write_mongo_to_h5()


# ~ 8m 12s , 8.7G
# write_mongo_to_pickle()


# data_pipe = build_datapipes_pickle()
# data_pipe = build_datapipes("train*")


#
#
# loader = torch.utils.data.DataLoader(
#         dataset=data_pipe,
#         batch_size=1,
#         shuffle=True,
#         drop_last=True,
#         num_workers=0, collate_fn=lambda x: x)


#pickle:
# 3m 21s for 1 worker, 2m 14 for 8, 58s for 0?

#h5: 20s for 0 worker
# tmp = 0
# for data in tqdm(loader):
#     print (data[0][0].x)
#     break
# tmp += 32
#
# print (tmp)

# mongo: ~6m






















# def write_mongo_to_pickle():
#     data_module = MongoDataModule(config={'buf_size': 2048, 'batch_size': 32, 'db_name': 'hol_step',
#                                                          'collection_name': 'pretrain_graphs',
#                                                          'options': ['edge_attr', 'edge_index', 'softmax_idx']})
#
#     data_module.setup('fit')
#     data_module.setup('test')
#
#     train_loader = iter(data_module.train_dataloader())
#     val_loader = iter(data_module.val_dataloader())
#     test_loader = iter(data_module.test_dataloader())
#
#
#     BATCHES_PER_FILE = 512
#
#
#     data_list = []
#     file_num = 0
#     for i, batch in tqdm(enumerate(train_loader)):
#         data_list.append(batch)
#
#         if i > 0 and i % BATCHES_PER_FILE == 0:
#             to_pickle(data_list, data_dir + f'/{file_num}.pk')
#             file_num += 1
#             data_list = []


# @torchdata.datapipes.functional_datapipe("load_pickle_data")
# class PickleDataLoader(torchdata.datapipes.iter.IterDataPipe):
#     def __init__(self, source_datapipe, **kwargs) -> None:
#         self.source_datapipe = source_datapipe
#         # self.transform = kwargs['transform']
#
#     def __iter__(self):
#
#         for file_name in self.source_datapipe:
#
#             with open(file_name, 'rb') as pkl_file:
#                 data_list = pickle.load(pkl_file)
#
#                 for data in data_list:
#                     yield data
#
# def build_pickle_datapipe() -> torchdata.datapipes.iter.IterDataPipe:
#     datapipe = torchdata.datapipes.iter.FileLister(data_dir, masks='*.pk')
#     datapipe = datapipe.shuffle()
#     datapipe = datapipe.sharding_filter()
#     datapipe = datapipe.load_pickle_data()
#     return datapipe

