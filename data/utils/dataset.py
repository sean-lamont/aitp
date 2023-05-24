import torchdata
import os
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
                ret.attention_edge_index_t = torch.cartesian_prod(torch.arange(x1_mat.size(0)),
                                                                        torch.arange(x1_mat.size(0))).transpose(0,1)

                ret.attention_edge_index_s = torch.cartesian_prod(torch.arange(x2_mat.size(0)),
                                                                        torch.arange(x2_mat.size(0))).transpose(0,1)
                # raise NotImplementedError

        if 'softmax_idx' in options:
            ret.softmax_idx_t = x1_edge_index.size(1)
            ret.softmax_idx_s = x2_edge_index.size(1)

        return ret

    def custom_collate(self, data):
        data_list = [self.sample_to_link(d) for d in data]
        batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_t'])
        return separate_link_batch(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # batch = batch[0]
        data_1, data_2, y = batch

        data_1.x = data_1.x.to(device)
        data_2.x = data_2.x.to(device)

        data_1.edge_index = data_1.edge_index.to(device)
        data_2.edge_index = data_2.edge_index.to(device)

        data_1.edge_attr = data_1.edge_attr.to(device)
        data_2.edge_attr = data_2.edge_attr.to(device)

        y = y.to(device)
        # batch.edge_index = batch.edge_index.to(device)
        return data_1, data_2, y

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
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate,
                                           num_workers=0)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)


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


# data_dir = '/home/sean/Documents/phd/repo/aitp/data/utils/holstep_full'

# save preprocessed graph data in batches to files. Prevents need for recomputing batch level attributes,
# such as ptr, batch idx etc.
def write_mongo_to_h5(data_dir):
    data_module = MongoDataModule(config={'buf_size': 2048, 'batch_size': 32, 'db_name': 'hol_step',
                                                         'collection_name': 'pretrain_graphs',
                                                         'options': ['edge_attr', 'edge_index', 'softmax_idx',
                                                                     'attention_edge_index']})

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
    def __init__(self, source_datapipe, cycle=False,**kwargs) -> None:
        if cycle:
            self.source_datapipe = source_datapipe.cycle()
        else:
            self.source_datapipe = source_datapipe
        # self.transform = kwargs['transform']
    # def file_to_data(self, h5file):

    def __iter__(self):
        for file_name in self.source_datapipe:
            with h5py.File(file_name, 'r') as h5file:
                x1 = torch.from_numpy(h5file['data/x1'][:])
                edge_index1 = torch.from_numpy(h5file['data/edge_index1'][:])#.long()
                edge_attr1 = torch.from_numpy(h5file['data/edge_attr1'][:])#.long()
                batch1 = torch.from_numpy(h5file['data/batch1'][:])#.long()

                # ptr1 = torch.from_numpy(h5file['data/ptr1'][:]).tolist()
                ptr1 = torch.from_numpy(h5file['data/ptr1'][:])#.long()

                edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:]).tolist()
                # edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:]).long()

                # attention_edge_1 = torch.from_numpy(h5file['data/attention_edge1'][:])

                x2 = torch.from_numpy(h5file['data/x2'][:])#.long()
                edge_index2 = torch.from_numpy(h5file['data/edge_index2'][:])#.long()
                edge_attr2 = torch.from_numpy(h5file['data/edge_attr2'][:])#.long()
                batch2 = torch.from_numpy(h5file['data/batch2'][:])#.long()

                # ptr2 = torch.from_numpy(h5file['data/ptr2'][:]).tolist()
                ptr2 = torch.from_numpy(h5file['data/ptr2'][:])#.long()

                edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:]).tolist()
                # edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:]).long()

                # attention_edge_2 = torch.from_numpy(h5file['data/attention_edge2'][:])

                y = torch.from_numpy(h5file['data/y'][:])#.long()

                data_len_1 = torch.from_numpy(h5file['data/x1_len'][:])
                data_len_2 = torch.from_numpy(h5file['data/x2_len'][:])

                for i in range(len(x1)):
                    num_nodes1 = data_len_1[i][0]
                    num_edges1 = data_len_1[i][1]

                    num_nodes2 = data_len_2[i][0]
                    num_edges2 = data_len_2[i][1]

                    data_1 = Data(x=x1[i, :num_nodes1],
                               edge_index=edge_index1[i, :, :num_edges1],
                               edge_attr=edge_attr1[i, :num_edges1],
                               batch=batch1[i, :num_nodes1],
                               ptr=ptr1[i],
                               # attention_edge_index=attention_edge_1[i],
                               softmax_idx=edge_ptr1[i])

                    data_2 = Data(x=x2[i, :num_nodes2],
                             edge_index=edge_index2[i, :, :num_edges2],
                             edge_attr=edge_attr2[i, :num_edges2],
                             batch=batch2[i, :num_nodes2],
                             ptr=ptr2[i],
                            # attention_edge_index=attention_edge_2[i],
                            softmax_idx=edge_ptr2[i])

                    y_i = y[i]

                    yield data_1, data_2, y_i

def build_h5_datapipe(masks, data_dir, cycle=False):
    datapipe = torchdata.datapipes.iter.FileLister(data_dir, masks=masks)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.load_h5_data(cycle=cycle)
    return datapipe

def collate_to_relation(batch):
    data_1, data_2, y = batch[0]

    x = data_1.x + 1
    edge_index = data_1.edge_index

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])


    xi = torch.tensor_split(xi, data_1.softmax_idx[1:-1])
    xj = torch.tensor_split(xj, data_1.softmax_idx[1:-1])


    edge_attr_ = data_1.edge_attr.long()
    edge_attr_ = torch.tensor_split(edge_attr_, data_1.softmax_idx[1:-1])
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attr_)


    xi = torch.nn.utils.rnn.pad_sequence(xi)
    xj = torch.nn.utils.rnn.pad_sequence(xj)

    xi = xi[:300]
    xj = xj[:300]
    edge_attr_ = edge_attr_[:300]

    # mask = torch.stack([torch.cat([xi[:, i] != 0, torch.BoolTensor([True])], dim=0) for i in range(xi.shape[1])], dim = 0)

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    data_1 = Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)

    x = data_2.x + 1
    edge_index = data_2.edge_index

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    xi = torch.tensor_split(xi, data_2.softmax_idx[1:-1])
    xj = torch.tensor_split(xj, data_2.softmax_idx[1:-1])

    edge_attr_ = data_2.edge_attr.long()
    edge_attr_ = torch.tensor_split(edge_attr_, data_2.softmax_idx[1:-1])
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attr_)

    xi = torch.nn.utils.rnn.pad_sequence(xi)
    xj = torch.nn.utils.rnn.pad_sequence(xj)

    xi = xi[:300]
    xj = xj[:300]
    edge_attr_ = edge_attr_[:300]


    # mask = torch.stack([torch.cat([xi[:, i] != 0, torch.BoolTensor([True,])], dim=0) for i in range(xi.shape[1])], dim = 0)

    # mask = (xi != 0).T
    # mask = None

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    data_2 = Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)

    return data_1, data_2, y



class H5DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['data_dir']

    def prepare_data(self):
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
            write_mongo_to_h5(self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_pipe = build_h5_datapipe("train*", self.data_dir)
            self.val_pipe = build_h5_datapipe("val*",self.data_dir, cycle=True)
        if stage == "test":
            self.test_pipe = build_h5_datapipe("test*", self.data_dir)

    def train_dataloader(self):
         return torch.utils.data.DataLoader(
            dataset=self.train_pipe,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            # pin_memory=True,
            num_workers=0, collate_fn=lambda x: x)
            # num_workers = 4, collate_fn = collate_to_relation)

    def val_dataloader(self):
        # cycle through
        return torch.utils.data.DataLoader(
            dataset=self.val_pipe,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0, collate_fn=lambda x: x)
            # num_workers = 0, collate_fn = collate_to_relation)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_pipe,
            batch_size=1,
            # shuffle=True,
            drop_last=True,
            num_workers=0, collate_fn=lambda x: x)
            # num_workers = 0, collate_fn = collate_to_relation)

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     data_1, data_2, y = batch
    #     data_1 = data_1.to(device)
    #     data_2= data_2.to(device)
    #     y = y.to(device)
    #     return data_1, data_2, y
    #
    # only transfer relevant properties to device
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = batch[0]
        data_1, data_2, y = batch

        data_1.x = data_1.x.to(device)
        data_2.x = data_2.x.to(device)

        data_1.edge_index = data_1.edge_index.to(device).long()
        data_2.edge_index = data_2.edge_index.to(device).long()

        data_1.edge_attr = data_1.edge_attr.to(device).long()
        data_2.edge_attr = data_2.edge_attr.to(device).long()

        data_1.batch = data_1.batch.to(device).long()
        data_2.batch = data_2.batch.to(device).long()

        data_1.ptr = data_1.ptr.to(device).long()
        data_2.ptr = data_2.ptr.to(device).long()

        # data_1.softmax_idx = data_1.softmax_idx.to(device)
        # data_2.softmax_idx = data_2.softmax_idx.to(device)

        if hasattr(data_1, "attention_edge_index"):
            data_1.attention_edge_index = data_1.attention_edge_index.to(device)
            data_2.attention_edge_index = data_2.attention_edge_index.to(device)

        y = y.to(device)
        # batch.edge_index = batch.edge_index.to(device)
        return data_1, data_2, y


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index





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

# class MaskData(Data):
#     def __init__(self,x,edge_index=None,edge_attr=None,softmax_idx=None,mask_idx=None,mask_raw=None):
#         super().__init__()
#         self.x = x
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr
#         self.sofmax_idx = softmax_idx
#         self.mask_idx = mask_idx
#         self.mask_raw = mask_raw
#
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'edge_index':
#             return self.x.size(0)
#         elif key == 'softmax_idx':
#             return self.softmax_idx
#         elif key == 'mask_idx':
#             return self.x.size(0)
#
#         return super().__inc__(key, value, *args, **kwargs)
#
#
# redundant
# def ptr_to_complete_edge_index(ptr):
#     from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
#     to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
#     combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
#     return combined_complete_edge_index





# def to_batch_graph(batch, graph_collection, options):
#     batch_list = []
#
#
#     stmts = list(set([sample['stmt'] for sample in batch]))
#     conjs = list(set([sample['conj'] for sample in batch]))
#
#     stmts.extend(conjs)
#
#     exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))
#
#     expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}
#
#
#     for sample in batch:
#
#         stmt = sample['stmt']
#         conj = sample['conj']
#         y = sample['y']
#
#         conj_graph = expr_dict[conj]
#         stmt_graph = expr_dict[stmt]
#
#         x1 = conj_graph['onehot']
#         x1_mat = torch.LongTensor(x1)
#
#         x2 = stmt_graph['onehot']
#         x2_mat = torch.LongTensor(x2)
#
#         tmp_batch = LinkData(x_s=x2_mat, x_t=x1_mat, y=torch.tensor(y))
#
#
#         if 'edge_index' in options:
#             if 'edge_index' in conj_graph and 'edge_index' in stmt_graph:
#                 x1_edge_index = conj_graph['edge_index']
#                 x1_edge_index = torch.LongTensor(x1_edge_index)
#
#                 x2_edge_index = stmt_graph['edge_index']
#                 x2_edge_index = torch.LongTensor(x2_edge_index)
#
#                 tmp_batch.edge_index_t = x1_edge_index
#                 tmp_batch.edge_index_s = x2_edge_index
#             else:
#                 raise NotImplementedError
#
#
#         if 'edge_attr' in options:
#             if 'edge_attr' in conj_graph and 'edge_attr' in stmt_graph:
#                 x1_edge_attr = conj_graph['edge_attr']
#                 x1_edge_attr = torch.LongTensor(x1_edge_attr)
#
#                 x2_edge_attr = stmt_graph['edge_attr']
#                 x2_edge_attr = torch.LongTensor(x2_edge_attr)
#
#                 tmp_batch.edge_attr_t = x1_edge_attr
#                 tmp_batch.edge_attr_s = x2_edge_attr
#             else:
#                 raise NotImplementedError
#
#         # Edge index used to determine where attention is propagated in Message Passing Attention schemes
#
#         if 'attention_edge_index' in options:
#             if 'attention_edge_index' in conj_graph and 'attention_edge_index' in stmt_graph:
#                 tmp_batch.attention_edge_index_t = conj_graph['attention_edge_index']
#                 tmp_batch.attention_edge_index_s = stmt_graph['attention_edge_index']
#             else:
#                 # Default is global attention
#                 tmp_batch.attention_edge_index_t = torch.cartesian_prod(torch.arange(x1_mat.size(0)),
#                                                                         torch.arange(x1_mat.size(0))).transpose(0,1)
#
#                 tmp_batch.attention_edge_index_s = torch.cartesian_prod(torch.arange(x2_mat.size(0)),
#                                                                         torch.arange(x2_mat.size(0))).transpose(0,1)
#
#
#         #todo make data options have possible values i.e. options['softmax_idx'] == AMR, use edges, else directed attention etc.
#         if 'softmax_idx' in options:
#             tmp_batch.softmax_idx_t = x1_edge_index.size(1)
#             tmp_batch.softmax_idx_s = x2_edge_index.size(1)
#
#
#         # todo positional encoding including with depth
#
#
#         batch_list.append(tmp_batch)
#
#
#     loader = iter(DataLoader(batch_list, batch_size=len(batch_list), follow_batch=['x_s', 'x_t']))
#
#     batch_ = next(iter(loader))
#
#     g,p,y = separate_link_batch(batch_, options)
#
#     return g,p,y



# def to_batch_transformer(batch, graph_collection, options):
#
#     stmts = list(set([sample['stmt'] for sample in batch]))
#     conjs = list(set([sample['conj'] for sample in batch]))
#
#     stmts.extend(conjs)
#
#     exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))
#
#     expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}
#
#     # just use CLS token as separate (add 4 to everything)
#     word_dict = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}
#
#     #start of sentence is CLS
#     conj_X = [torch.LongTensor([-3] + expr_dict[sample['conj']]['onehot']) + 4 for sample in batch]
#     stmt_X = [torch.LongTensor([-3] + expr_dict[sample['stmt']]['onehot']) + 4 for sample in batch]
#
#     Y = torch.LongTensor([sample['y'] for sample in batch])
#
#     # batch_list = pad_sequence(conj_X), pad_sequence(stmt_X), Y
#
#     # loader = iter(StandardLoader(batch_list), batch_size = len(batch))
#
#     # batch_ = next(iter(loader))
#
#
#     return pad_sequence(conj_X), pad_sequence(stmt_X), Y
#
#
#
#
#
#
# def test_iter(batches, batch_fn, graph_collection, options):
#     for batch in batches:
#         yield batch_fn(batch, graph_collection, options)
#
#
# def val_iter(batches, batch_fn, graph_collection, options):
#     for batch in itertools.cycle(batches):
#         yield batch_fn(batch, graph_collection, options)
#


# def to_masked_batch_graph(graphs, options):
#     batch_list = []
#
#     for graph in graphs:
#         graph = graph['graph']
#         x =  graph['onehot']
#
#         # todo mask edges as well
#         # mask some tokens
#
#         masking_rate = 0.15
#
#         x = torch.LongTensor(x)
#
#         #add 1 for mask
#         x = x + 1
#
#
#         mask = torch.rand(x.size(0)).ge(masking_rate)
#
#         y = torch.index_select(x, (x * mask == 0).nonzero().flatten())
#
#         x = x * mask
#         mask_inds = (x == 0).nonzero().flatten()
#
#         #need to set y as the values
#
#         tmp_batch = MaskData(x = x)
#
#         tmp_batch.y = y
#
#         if 'edge_index' in options:
#             if 'edge_index' in graph:
#                 tmp_batch.edge_index = torch.LongTensor(graph['edge_index'])
#             else:
#                 raise NotImplementedError
#
#         if 'edge_attr' in options:
#             if 'edge_attr' in graph:
#                 tmp_batch.edge_attr = torch.LongTensor(graph['edge_attr'])
#             else:
#                 raise NotImplementedError
#
#         if 'attention_edge_index' in options:
#             if 'attention_edge_index' in graph:
#                 tmp_batch.attention_edge_index = graph['attention_edge_index']
#             else:
#                 # Default is global attention
#                 tmp_batch.attention_edge_index = torch.cartesian_prod(torch.arange(x.size(0)),
#                                                                       torch.arange(x.size(0))).transpose(0,1)
#
#
#         #todo make data options have possible values i.e. options['softmax_idx'] == AMR, use edges, else directed attention etc.
#         if 'softmax_idx' in options:
#             tmp_batch.softmax_idx = len(graph['edge_index'][0])
#
#         tmp_batch.mask_idx = mask_inds
#         tmp_batch.mask_raw = mask_inds
#         # todo positional encoding including with depth
#
#         batch_list.append(tmp_batch)
#
#
#     loader = iter(DataLoader(batch_list, batch_size=len(batch_list)))
#
#     batch_ = next(iter(loader))
#     print (batch_.softmax_idx.size(0))
#
#     return batch_

# def get_data(config):
#     if config['data_source'] == "MongoDB":
#         client = MongoClient()
#         db = client[config['dbname']]
#         graph_collection = db[config['graph_collection_name']]
#         split_collection = db[config['split_name']]
#         return graph_collection, split_collection
#     else:
#         return NotImplementedError
