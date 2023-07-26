"""
Utilities for graph data with Pytorch Geometric
"""
import torch.nn
from torch_geometric.data import Data, Batch
import torch
import torch_geometric
import logging

"""
DirectedData class, used for batches with attention_edge_index in SAT models
"""


class DirectedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'attention_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


'''
Function to generate a "complete_edge_index" given a ptr corresponding to a PyG batch.
 This is used in vanilla Structure Aware Attention (SAT) models with full attention.
'''


def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


'''
Utility functions for computing ancestor and descendant nodes and node depth for a PyG graph. 
Used for masking attention in Structure Aware Transformer (SAT) Models
'''


def get_directed_edge_index(num_nodes, edge_idx):
    if num_nodes == 1:
        return torch.LongTensor([[], []])

    from_idx = []
    to_idx = []

    for i in range(0, num_nodes):
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx)
        except Exception as e:
            logging.warning(f"Exception {e}, {i}, {edge_idx}, {num_nodes}")
            continue

        found_nodes = list(ancestor_nodes.numpy())

        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        try:
            children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx,
                                                                                  flow='target_to_source')
        except Exception as e:
            logging.warning(f"Exception {e}, {i}, {edge_idx}, {num_nodes}")
            continue

        found_nodes = list(children_nodes.numpy())

        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


def get_depth_from_graph(num_nodes, edge_index):
    to_idx = edge_index[1]

    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i,
                                                                          edge_index=edge_index,
                                                                          flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes

    return depths


def list_to_sequence(data_list, max_len):
    data_list = torch.nn.utils.rnn.pad_sequence(data_list)
    data_list = data_list[:max_len]
    mask = (data_list == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)
    return (data_list, mask)


def list_to_relation(data_list, max_len):
    xis = [d[0] for d in data_list][:max_len]
    xjs = [d[1] for d in data_list][:max_len]
    edge_attrs = [d[2] for d in data_list][:max_len]

    xi = torch.nn.utils.rnn.pad_sequence(xis)
    xj = torch.nn.utils.rnn.pad_sequence(xjs)
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attrs)

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    return Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)


def list_to_graph(data_list, attributes):
    data_list = Batch.from_data_list(data_list)
    if 'attention_edge' in attributes and attributes['attention_edge'] == 'full':
        data_list.attention_edge_index = ptr_to_complete_edge_index(data_list.ptr)
    return data_list


def to_data(expr, data_type, vocab, config=None):
    if data_type == 'graph':
        data = DirectedData(x=torch.LongTensor([vocab[a] if a in vocab else vocab['UNK'] for a in expr['tokens']]),
                            edge_index=torch.LongTensor(expr['edge_index']),
                            edge_attr=torch.LongTensor(expr['edge_attr']), )

        if config:
            if 'attention_edge' in config.attributes and config.attributes['attention_edge'] == 'directed':
                data.attention_edge_index = torch.LongTensor(expr['attention_edge_index'])
            if 'pe' in config.attributes:
                data.abs_pe = torch.LongTensor(expr[config.attributes['pe']])

        return data

    elif data_type == 'sequence':
        return torch.LongTensor([vocab[a] if a in vocab else vocab['UNK'] for a in expr['full_tokens']])

    elif data_type == 'fixed':
        return expr

    elif data_type == 'relation':
        x = [vocab[a] if a in vocab else vocab['UNK'] for a in expr['tokens']]
        edge_index = expr['edge_index']
        edge_attr = torch.LongTensor(expr['edge_attr'])
        xi = torch.LongTensor([x[i] for i in edge_index[0]])
        xj = torch.LongTensor([x[i] for i in edge_index[1]])
        return (xi, xj, edge_attr)


def list_to_data(batch, config):
    if config.type == 'graph':
        return list_to_graph(batch, config.attributes)
    elif config.type == 'sequence':
        return list_to_sequence(batch, config.attributes['max_len'])
    elif config.type == 'relation':
        return list_to_relation(batch, config.attributes['max_len'])
    elif config.type == 'fixed':
        return batch
