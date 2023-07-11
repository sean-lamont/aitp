import torch
from multiprocessing import Pool
from tqdm import tqdm
import torch_geometric
from pymongo import MongoClient


def get_directed_edge_index(num_nodes, edge_idx):
    if num_nodes == 1:
        return torch.LongTensor([[],[]])

    from_idx = []
    to_idx = []

    for i in range(0, num_nodes):
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx)
        except Exception as e:
            print (f"exception {e}, {i}, {edge_idx}, {num_nodes}")
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
            print (f"exception {e}, {i}, {edge_idx}, {num_nodes}")
            continue

        found_nodes = list(children_nodes.numpy())

        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)

# probably slow, could recursively do k-hop subgraph with k = 1 instead
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


if __name__ == '__main__':

    client = MongoClient()
    db = client['hol_step']
    expr_collection = db['expression_graphs']

    def update_attention_func(item):
        expr_collection.update_many({"_id": item["_id"]},
                                    {"$set":
                                        {
                                            "graph.attention_edge_index":
                                                get_directed_edge_index(len(item['graph']['onehot']),
                                                                        torch.LongTensor(
                                                                            item['graph']['edge_index'])).tolist(),
                                            "graph.depth":
                                                get_depth_from_graph(len(item['graph']['onehot']),
                                                                     torch.LongTensor(
                                                                         item['graph']['edge_index'])).tolist()
                                        }})

    items = []
    for item in tqdm(expr_collection.find({})):
        items.append(item)

    pool = Pool(processes=3)
    for _ in tqdm.tqdm(pool.imap_unordered(update_attention_func, items), total=len(items)):
        pass


