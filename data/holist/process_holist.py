import logging
import re
from multiprocessing import Pool

from data.utils.graph_data_utils import get_depth_from_graph, get_directed_edge_index
from pymongo import MongoClient
import os

import torch
from tqdm import tqdm

from experiments.holist import io_util, deephol_pb2
from experiments.holist.deephol_loop import options_pb2
from experiments.holist.utilities.sexpression_graphs import SExpressionGraph
from experiments.holist.utilities.sexpression_to_graph import sexpression_to_graph
from experiments.holist.utilities import prooflog_to_examples


def tokenize_string(string):
    pattern = r'(\(|\)|\s)'
    tokens = re.split(pattern, string)
    tokens = [token for token in tokens if token.strip()]  # Remove empty tokens
    return tokens


def sexpression_to_polish(sexpression_text):
    sexpression = SExpressionGraph()
    sexpression.add_sexp(sexpression_text)
    out = []

    def process_node(node):
        if len(sexpression.get_children(node)) == 0:
            out.append(node)
        for i, child in enumerate(sexpression.get_children(node)):
            if i == 0:
                out.append(sexpression.to_text(child))
                continue
            process_node(sexpression.to_text(child))

    process_node(sexpression.to_text(sexpression.roots()[0]))
    return out


# gen vocab dictionary from file
def gen_vocab_dict(vocab_file):
    with open(vocab_file) as f:
        x = f.readlines()
    vocab = {}
    for a, b in enumerate(x):
        vocab[b.replace("\n", "")] = a
    return vocab

def prepare_data(config):
    tac_dir = config['tac_dir']
    theorem_dir = config['theorem_dir']
    train_logs = config['train_logs']
    val_logs = config['val_logs']
    vocab_file = config['vocab_file']
    source = config['source']
    data_options = config['data_options']

    logging.info('Generating data..')

    scrub_parameters = options_pb2.ConvertorOptions.NOTHING

    logging.info('Loading theorem database..')
    theorem_db = io_util.load_theorem_database_from_file(theorem_dir)

    train_logs = io_util.read_protos(train_logs, deephol_pb2.ProofLog)
    val_logs = io_util.read_protos(val_logs, deephol_pb2.ProofLog)

    options = options_pb2.ConvertorOptions(tactics_path=tac_dir, scrub_parameters=scrub_parameters)
    converter = prooflog_to_examples.create_processor(options=options, theorem_database=theorem_db)

    logging.info('Loading proof logs..')
    train_proof_logs = []
    for j, i in tqdm(enumerate(converter.process_proof_logs(train_logs))):
        train_proof_logs.append(i)

    val_proof_logs = []
    for j, i in tqdm(enumerate(converter.process_proof_logs(val_logs))):
        val_proof_logs.append(i)

    train_params = []
    val_params = []
    for a in train_proof_logs:
        train_params.extend(a['thms'])
    for a in val_proof_logs:
        val_params.extend(a['thms'])

    all_params = train_params + val_params

    all_exprs = list(
        set([a['goal'] for a in train_proof_logs] + [a['goal'] for a in val_proof_logs] + all_params))

    logging.info(f'{len(all_exprs)} unique expressions')
    logging.info('Generating data dictionary from expressions..')

    expr_dict = {expr: sexpression_to_graph(expr) for expr in tqdm(all_exprs)}

    train_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id'],
                         'thms_hard_negatives': a['thms_hard_negatives'], 'split': 'train'} for a in
                        train_proof_logs]

    val_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id'],
                       'thms_hard_negatives': a['thms_hard_negatives'], 'split': 'val'} for a in val_proof_logs]

    if vocab_file:
        logging.info(f'Generating vocab from file {vocab_file}..')
        vocab = gen_vocab_dict(vocab_file)
        vocab['UNK'] = len(vocab)

    else:
        logging.info(f'Generating vocab from proof logs..')

        vocab_toks = set([token for expr in tqdm(expr_dict.values()) for token in expr['tokens']])
        vocab = {}
        for i, v in enumerate(vocab_toks):
            vocab[v] = i + 1

        vocab["UNK"] = len(vocab)

    if source == 'mongodb':
        logging.info("Adding data to MongoDB")
        client = MongoClient()
        db = client[data_options['db']]
        expr_col = db['expression_graphs']
        split_col = db['split_data']
        vocab_col = db['vocab']
        thm_ls_col = db['train_thm_ls']

        logging.info("Adding full expression data..")

        for k, v in tqdm(expr_dict.items()):

            expr_col.insert_one({'_id': k, 'data': {'tokens': v['tokens'],
                                                    'edge_index': v['edge_index'],
                                                    'edge_attr': v['edge_attr'],}})

        split_col.insert_many(train_proof_logs)
        split_col.insert_many(val_proof_logs)

        vocab_col.insert_many([{'_id': k, 'index': v} for k, v in vocab.items()])

        thm_ls_col.insert_many([{'_id': x} for x in list(set(train_params))])


    elif source == 'directory':
        data = {'train_data': train_proof_logs, 'val_data': val_proof_logs,
                'expr_dict': expr_dict, 'train_thm_ls': list(set(train_params)), 'vocab': vocab}

        save_dir = data_options['dir']
        os.makedirs(save_dir)
        torch.save(data, save_dir + '/data.pt')

    else:
        raise NotImplementedError


    def add_addional_data(item):
        expr_col.update_many({"_id": item["_id"]},
                                    {"$set":
                                        {
                                            # "data.attention_edge_index":
                                            #     get_directed_edge_index(len(item['graph']['tokens']),
                                            #                             torch.LongTensor(
                                            #                                 item['graph']['edge_index'])).tolist(),
                                            # "data.depth":
                                            #     get_depth_from_graph(len(item['graph']['tokens']),
                                            #                          torch.LongTensor(
                                            #                              item['graph']['edge_index'])).tolist(),

                                            'data.full_tokens': tokenize_string(item["_id"]),
                                            'data.polished': sexpression_to_polish(item["_id"])
                                        }})

    if config['additional_data']:
        assert source == 'mongodb', "Currently only MongoDB is supported for HOList additional fields"

        logging.info("Adding additional properties to expression database..")
        items = []

        for item in tqdm(expr_col.find({})):
            items.append(item)

        pool = Pool(processes=4)
        for _ in tqdm(pool.imap_unordered(add_addional_data, items), total=len(items)):
            pass

    logging.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # save_dir = '/home/sean/Documents/phd/deepmath-light/deepmath/combined_train_data'

    # vocab_file = '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/vocab_ls.txt'

    human_train_logs = 'data/holist/raw_data/hollightdata/final/proofs/human/train/prooflogs*'
    human_val_logs = 'data/holist/raw_data/hollightdata/final/proofs/human/valid/prooflogs*'

    synthetic_train_logs = 'data/holist/raw_data/hollightdata/final/proofs/synthetic/train/prooflogs*'

    all_train_logs = synthetic_train_logs + ',' + human_train_logs
    all_val_logs = human_val_logs

    config = {
        'tac_dir': 'data/holist/hollight_tactics.textpb',
        'theorem_dir': 'data/holist/theorem_database_v1.1.textpb',
        'train_logs': human_train_logs,
        'val_logs': human_val_logs,
        'vocab_file': None,
        'source': 'mongodb',
        'data_options': {'db': 'holist'},
        'additional_data': True
    }

    prepare_data(config)

