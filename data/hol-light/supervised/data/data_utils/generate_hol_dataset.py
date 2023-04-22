import os
import data_loader
from pymongo import MongoClient
import sys
import argparse
import pickle
import random
from tqdm import tqdm

import torch

from holstep_parser import graph_from_hol_stmt
from holstep_parser import tree_from_hol_stmt


def generate_dataset(path, converter, split, graph_collection, split_collection,files=None):

    if files is None:
        files = os.listdir(path)


    # print (os.curdir)
    loader = data_loader.DataLoader("../raw_data/raw_data/train", "../dicts/hol_train_dict")

    for i in tqdm(range(len(files))):
        fname = files[i]
        fpath = os.path.join(path, fname)
        # print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))

        with open(fpath, 'r') as f:
            next(f)
            conj_symbol = next(f)
            conj_token = next(f)
            assert conj_symbol[0] == 'C'

            if not graph_collection.find_one({"_id": conj_symbol[2:]}):
                conjecture = converter(conj_symbol[2:], conj_token[2:])

                onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(conjecture)

                edge_index = torch.stack([oindex1, oindex2], dim=0).long().tolist()

                info = graph_collection.insert_one({"_id" : conj_symbol[2:], "graph": {"onehot": onehot.tolist(), "edge_index": edge_index, "edge_attr": edge_attr}})

            for line in f:
                if line and line[0] in '+-':

                    stmt_symbol = line[2:]
                    stmt_token = next(f)[2:]

                    # if stmt_symbol not in expressions:
                    if not graph_collection.find_one({"_id": stmt_symbol}):

                        statement = converter(stmt_symbol, stmt_token)

                        onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(statement)

                        edge_index = torch.stack([oindex1, oindex2], dim=0).long().tolist()

                        info = graph_collection.insert_one({"_id" : stmt_symbol, "graph": {"onehot": onehot.tolist(), "edge_index": edge_index, "edge_attr": edge_attr}})


                    flag = 1 if line[0] == '+' else 0

                    info = split_collection.insert_one({"flag":flag, "conj":conj_symbol[2:], "stmt":stmt_symbol, "split": split})



if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(
        description='Generate graph dataset from HolStep')

    parser.add_argument('path', type=str, help='Path to the root of HolStep dataset')

    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree of graph (default).')

    parser.add_argument(
        '--db_name',
        type=str,
        default='hol_step',
        help='Name of MongoDB database')


    parser.add_argument(
        '--graph_name',
        type=str,
        default='expression_graphs',
        help='Graph Expression collection name for database')


    parser.add_argument(
        '--split_name',
        type=str,
        default='train_val_test_data',
        help='Train, Val, Test split collection name for database')

    args = parser.parse_args()

    format_choice = {
        'graph': lambda x, y: graph_from_hol_stmt(x, y),
        'tree': lambda x, y: tree_from_hol_stmt(x, y)}

    print (args.path)

    assert os.path.isdir(args.path), 'Saving path must be a folder'

    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')
    valid_path = os.path.join(args.path, 'valid')

    print (train_path)

    files = os.listdir(train_path)

    valid_files = random.sample(files, int(len(files)*0.07+0.5))
    train_files = [x for x in files if x not in valid_files]


    db_client = MongoClient()
    db_name = args.db_name
    graph_name = args.graph_name
    split_name = args.split_name
    db = db_client[db_name]


    graph_collection = db[graph_name]
    split_collection = db[split_name]

    print (f"Saving to MongoDB database {db_name}, graph representations to collection {graph_name} and train/test/val pairs to {split_name}")

    generate_dataset(train_path, format_choice[args.format],  split="train", graph_collection=graph_collection, split_collection=split_collection, files=train_files)

    generate_dataset(test_path, format_choice[args.format], split="test", graph_collection=graph_collection, split_collection=split_collection, files=None)

    generate_dataset(train_path, format_choice[args.format], split="valid", graph_collection=graph_collection, split_collection=split_collection, files=valid_files)
