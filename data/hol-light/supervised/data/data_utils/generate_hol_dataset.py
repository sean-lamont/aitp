import os
import sys
import argparse
import pickle
import random
from tqdm import tqdm

import torch

from holstep_parser import graph_from_hol_stmt
from holstep_parser import tree_from_hol_stmt

from pymongo import MongoClient
client = MongoClient()
db = client.hol_step

def generate_dataset(path, converter, split,files=None):
    '''Generate dataset at given path

    Parameters
    ----------
    path : str
        Path to the source
    output : str
        Path to the destination
    partition : int
        Number of the partition for this dataset (i.e. # of files)
    split : str
        Train, val or test
    '''

    if files is None:
        files = os.listdir(path)

    import data_loader

    print (os.curdir)
    loader = data_loader.DataLoader("../raw_data/raw_data/train", "../dicts/hol_train_dict")

    #mongodb collections for expression -> Graph dict and training data

    graph_collection = db.expression_graph_db
    split_collection = db.pairs


    for i, fname in enumerate(files):
        fpath = os.path.join(path, fname)
        print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))

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
        description='Generate graph repr dataset from HolStep')

    parser.add_argument('path', type=str, help='Path to the root of HolStep dataset')

    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree of graph (default).')

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

    generate_dataset(train_path, format_choice[args.format],  split="train",files=train_files)

    generate_dataset(test_path, format_choice[args.format], split="test", files=None)

    generate_dataset(train_path, format_choice[args.format], split="valid",files=valid_files)
