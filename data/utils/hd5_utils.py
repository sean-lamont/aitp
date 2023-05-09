import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class PremiseSelectionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
        # data1, data2, y = self.data_list[idx]
        # x1, edge_index1, edge_attr1, batch1, ptr1 = self._extract_data(data1)
        # x2, edge_index2, edge_attr2, batch2, ptr2 = self._extract_data(data2)
        #
        # return Data(x1, edge_index1, edge_attr1, batch1, ptr1), Data(x2, edge_index2, edge_attr2, batch2, ptr2), y

    # def _extract_data(self, data):
    #     x = data.x
    #     edge_index = data.edge_index
    #     edge_attr = data.edge_attr
    #     batch = data.batch
    #     ptr = data.ptr
    #
    #     return x, edge_index, edge_attr, batch, ptr

def to_hdf5(data_list: list, h5file: h5py.File, name: str):
    max_length = max(max(len(data1.x), len(data2.x)) for data1, data2, y in data_list)
    max_edge_length = max(max(len(data1.edge_index[0]), len(data2.edge_index[0])) for data1, data2, y in data_list)

    # len should be constant for ptr
    ptr_len = len(data_list[0][0].ptr)

    x1_dset = h5file.create_dataset(f"{name}/x1", shape=(len(data_list), max_length, data_list[0][0].num_node_features))
    edge_index1_dset = h5file.create_dataset(f"{name}/edge_index1", shape=(len(data_list), 2, max_edge_length), dtype=np.int64)
    edge_attr1_dset = h5file.create_dataset(f"{name}/edge_attr1", shape=(len(data_list), max_edge_length, data_list[0][0].num_edge_features))
    batch1_dset = h5file.create_dataset(f"{name}/batch1", shape=(len(data_list), max_length), dtype=np.int64)
    ptr1_dset = h5file.create_dataset(f"{name}/ptr1", shape=(len(data_list),ptr_len), dtype=np.int64)

    x2_dset = h5file.create_dataset(f"{name}/x2", shape=(len(data_list), max_length, data_list[0][1].num_node_features))
    edge_index2_dset = h5file.create_dataset(f"{name}/edge_index2", shape=(len(data_list), 2, max_edge_length), dtype=np.int64)
    edge_attr2_dset = h5file.create_dataset(f"{name}/edge_attr2", shape=(len(data_list), max_edge_length, data_list[0][1].num_edge_features))
    batch2_dset = h5file.create_dataset(f"{name}/batch2", shape=(len(data_list), max_length), dtype=np.int64)
    ptr2_dset = h5file.create_dataset(f"{name}/ptr2", shape=(len(data_list),ptr_len), dtype=np.int64)

    y_dset = h5file.create_dataset(f"{name}/y", shape=(len(data_list),ptr_len-1), dtype=np.float32)

    for i, (data1, data2, y) in enumerate(data_list):
        num_nodes1 = data1.num_nodes
        num_edges1 = data1.num_edges
        num_nodes2 = data2.num_nodes
        num_edges2 = data2.num_edges

    x1_dset[i, :num_nodes1] = data1.x.numpy().reshape(num_nodes1, 1)
    edge_index1_dset[i, :, :num_edges1] = data1.edge_index.numpy()
    edge_attr1_dset[i, :num_edges1] = data1.edge_attr.numpy().reshape(num_edges1, 1)
    batch1_dset[i, :num_nodes1] = data1.batch.numpy()#.reshape(num_nodes1, 1)
    ptr1_dset[i] = data1.ptr.numpy()

    x2_dset[i, :num_nodes2] = data2.x.numpy().reshape(num_nodes2, 1)
    edge_index2_dset[i, :, :num_edges2] = data2.edge_index.numpy()
    edge_attr2_dset[i, :num_edges2] = data2.edge_attr.numpy().reshape(num_edges2, 1)
    batch2_dset[i, :num_nodes2] = data2.batch.numpy()#.reshape(num_nodes2, 1)
    ptr2_dset[i] = data2.ptr.numpy()

    y_dset[i] = y.numpy()

# if __name__ == 'main':
#     data_list = [(data1, data2, y), ...] # a list of tuples where each tuple is (data, data, y)
#
#     with h5py.File('dataset.h5', 'w') as h5file:
#         to_hdf5(data_list, h5file, 'my_dataset')
#
# with h5py.File('dataset.h5', 'r') as h5file:
#     x1 = torch.from_numpy(h5file['my_dataset/x1'][:])
#     edge_index1 = torch.from_numpy(h5file['my_dataset/edge_index1'][:])
#     edge_attr1 = torch.from_numpy(h5file['my_dataset/edge_attr1'][:])
#     batch1 = torch.from_numpy(h5file['my_dataset/batch1'][:])
#     ptr1 = torch.from_numpy(h5file['my_dataset/ptr1'][:])
#
#     x2 = torch.from_numpy(h5file['my_dataset/x2'][:])
#     edge_index2 = torch.from_numpy(h5file['my_dataset/edge_index2'][:])
#     edge_attr2 = torch.from_numpy(h5file['my_dataset/edge_attr2'][:])
#     batch2 = torch.from_numpy(h5file['my_dataset/batch2'][:])
#     ptr2 = torch.from_numpy(h5file['my_dataset/ptr2'][:])
#
#     y = torch.from_numpy(h5file['my_dataset/y'][:])
#
# dataset = CustomDataset(data_list)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# for batch in loader:
#     # do something with the batch
#     pass
