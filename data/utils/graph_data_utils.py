"""
Utilities for graph data with Pytorch Geometric
"""
from torch_geometric.data import Data
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
