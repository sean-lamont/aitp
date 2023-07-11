import torch
from tqdm import tqdm
from pymongo import MongoClient
import pickle
from data.hol4.mongo_to_torch import get_depth_from_graph, get_directed_edge_index

if __name__ == '__main__':
    db = MongoClient()
    db = db['mizar']
    col = db['expression_graphs']

    with open("mizar_data.pk", "rb") as f:
        expr_dict = pickle.load(f)['expr_dict']

    for k,v in tqdm(expr_dict.items()):
        if not col.count_documents({'_id': k}, limit=1):
            v['attention_edge_index'] = get_directed_edge_index(len(v['tokens']), torch.LongTensor(v['edge_index'])).tolist()
            v['depth'] = get_depth_from_graph(len(v['tokens']), torch.LongTensor(v['edge_index'])).tolist()
            col.insert_one({'_id': k, 'graph': v})




