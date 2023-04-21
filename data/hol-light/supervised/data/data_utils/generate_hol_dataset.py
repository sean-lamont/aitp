import os
import sys
import argparse
import pickle
import random

import torch

from holstep_parser import graph_from_hol_stmt
from holstep_parser import tree_from_hol_stmt


def count_stmt(path):
    '''Count the number of the statements in the files of the given folder

    Parameters
    ----------
    path : str
        Path to the directory

    Returns
    -------
    int
        The number of total conjectures and statements
    '''
    total = 0
    files = os.listdir(path)
    for i, fname in enumerate(files):
        fpath = os.path.join(path, fname)
        print('Counting file {}/{} at {}.'.format(i + 1, len(files), fpath))
        with open(fpath, 'r') as f:
            total += sum([1 if line and line[0] in '+-C' else 0 for line in f])
    return total


def generate_dataset(path, output, partition, converter, files=None):
    '''Generate dataset at given path

    Parameters
    ----------
    path : str
        Path to the source
    output : str
        Path to the destination
    partition : int
        Number of the partition for this dataset (i.e. # of files)
    '''
    digits = len(str(partition))
    outputs = [[] for _ in range(partition)]
    if files is None:
        files = os.listdir(path)

    import data_loader
    from torch_geometric.data import Data

    print (os.curdir)
    loader = data_loader.DataLoader("../raw_data/raw_data/train", "../dicts/hol_train_dict")

    expressions = {}

    for i, fname in enumerate(files):
        fpath = os.path.join(path, fname)
        print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))
        with open(fpath, 'r') as f:
            next(f)
            conj_symbol = next(f)
            conj_token = next(f)
            assert conj_symbol[0] == 'C'

            if conj_symbol[2:] not in expressions:
                conjecture = converter(conj_symbol[2:], conj_token[2:])


                onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(conjecture)

                edge_index = torch.stack([oindex1, oindex2], dim=0).long()

                # nodes_one_hot = torch.nn.functional.one_hot(onehot, num_classes=1909).float()

                conj_graph = (onehot, edge_index, edge_attr)#Data(x=nodes_one_hot, edge_index=edge_index)

                expressions[conj_symbol[2:]]  = conj_graph


            for line in f:
                if line and line[0] in '+-':

                    stmt_symbol = line[2:]
                    stmt_token = next(f)[2:]

                    if stmt_symbol not in expressions:
                        statement = converter(stmt_symbol, stmt_token)


                        onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(statement)

                        edge_index = torch.stack([oindex1, oindex2], dim=0).long()

                        # nodes_one_hot = torch.nn.functional.one_hot(onehot, num_classes=1909).float()

                        stmt_graph = (onehot, edge_index, edge_attr)#Data(x=nodes_one_hot, edge_index=edge_index)

                        expressions[stmt_symbol] = stmt_graph

                    flag = 1 if line[0] == '+' else 0
                    # record = flag, conj_graph, stmt_graph, conj_symbol[2:], stmt_symbol #conjecture, statement
                    record = flag, conj_symbol[2:], stmt_symbol #conjecture, statement
                    # record = flag, conjecture, statement
                    outputs[random.randint(0, partition-1)].append(record)


    with open(os.path.join(output, 'expr_dict'),'wb') as f:
        pickle.dump(expressions, f)

    # Save dataset
    for i, data in enumerate(outputs):
        with open(
                os.path.join(output, 'holstep' + format(i, "0{}d".format(digits))),
                'wb') as f:
            print('Saving to file {}/{}'.format(i + 1, partition))
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(
        description='Generate graph repr dataset from HolStep')

    parser.add_argument('path', type=str, help='Path to the root of HolStep dataset')
    parser.add_argument('output', type=str, help='Output folder')
    parser.add_argument(
        '--train_partition',
        '-train',
        type=int,
        help='Number of the partition of the training dataset. Default=200',
        default=200)
    parser.add_argument(
        '--test_partition',
        '--test',
        type=int,
        help='Number of the partition of the testing dataset. Default=20',
        default=20)
    parser.add_argument(
        '--valid_partition',
        '--valid',
        type=int,
        help='Number of the partition of the validation dataset. Default=20',
        default=20)
    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree of graph (default).')

    args = parser.parse_args()

    format_choice = {
        'graph': lambda x, y: graph_from_hol_stmt(x, y),
        'tree': lambda x, y: tree_from_hol_stmt(x, y)}

    print (args.path, args.output)
    assert os.path.isdir(args.output), 'Data path must be a folder'
    assert os.path.isdir(args.path), 'Saving path must be a folder'
    train_output = os.path.join(args.output, 'train')
    test_output = os.path.join(args.output, 'test')
    valid_output = os.path.join(args.output, 'valid')
    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')
    valid_path = os.path.join(args.path, 'valid')
    print (train_path, train_output)
    if not os.path.exists(train_output):
        os.mkdir(train_output)
    if not os.path.exists(test_output):
        os.mkdir(test_output)
    files = os.listdir(train_path)
    valid_files = random.sample(files, int(len(files)*0.07+0.5))
    train_files = [x for x in files if x not in valid_files]
    print (valid_files)
    print (train_files)
    if not os.path.exists(valid_output):
        os.mkdir(valid_output)
    generate_dataset(train_path, train_output, args.train_partition,
                     format_choice[args.format], train_files)
    generate_dataset(test_path, test_output, args.test_partition,
                     format_choice[args.format])

    generate_dataset(train_path, valid_output, args.valid_partition,                     
            format_choice[args.format], valid_files)
