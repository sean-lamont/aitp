from multiprocessing import Pool
import torch
from pymongo import MongoClient
from tqdm import tqdm

from data.utils.graph_data_utils import get_directed_edge_index, get_depth_from_graph

# todo add raw holstep data, processing script and add all to mongo
if __name__ == '__main__':

    client = MongoClient()
    db = client['hol_step']
    expr_collection = db['expression_graphs']

    def update_attention_func(item):
        expr_collection.update_many({"_id": item["_id"]},
                                    {"$set":
                                        {
                                            "graph.attention_edge_index":
                                                get_directed_edge_index(len(item['graph']['tokens']),
                                                                        torch.LongTensor(
                                                                            item['graph']['edge_index'])).tolist(),
                                            "graph.depth":
                                                get_depth_from_graph(len(item['graph']['tokens']),
                                                                     torch.LongTensor(
                                                                         item['graph']['edge_index'])).tolist()
                                        }})

    items = []
    for item in tqdm(expr_collection.find({})):
        items.append(item)

    pool = Pool(processes=3)
    for _ in tqdm.tqdm(pool.imap_unordered(update_attention_func, items), total=len(items)):
        pass


