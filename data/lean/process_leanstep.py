import logging
import random
import re

from pymongo import MongoClient
from tqdm import tqdm

from experiments.holist.utilities.lean_sexpression_to_graph import sexpression_to_graph
from experiments.holist.utilities.sexpression_graphs import SExpressionGraph

if __name__ == '__main__':

    client = MongoClient()
    db = client['leanstep_sexpression']
    premise_classification = db['premise_classification']

    all_exprs = []

    data = [v for v in premise_classification.find()]

    RE = re.compile(" False| True")

    for d in data:
        assert d['classify_premise'].split()[-1] == 'False' or d['classify_premise'].split()[-1] == 'True'
        assert len(RE.split(d['classify_premise'])) == 2 and RE.split(d['classify_premise'])[1] == ''
        split = d['classify_premise'].split()
        y = 0 if split[-1] == 'False' else 1
        d['premise'] = RE.split(d['classify_premise'])[0]
        d['y'] = y

    # Proportion of negatives
    len([d['y'] for d in data if d['y'] == 0]) / len(data)

    data[0]

    goals = [d['goal'] for d in data]
    premises = [d['premise'] for d in data]

    all_exprs = set(set(goals) | set(premises))

    len(all_exprs)

    # add train_val_test data

    split = db['split_data']

    random.shuffle(data)

    # Train, val test split
    split.insert_many([{
        'conj': d['goal'],
        'stmt': d['premise'],
        'y': d['y'],
        'split': 'train'
    }
        for d in data[:int(0.8 * len(data))]])

    split.insert_many([{
        'conj': d['goal'],
        'stmt': d['premise'],
        'y': d['y'],
        'split': 'val'
    }
        for d in data[int(0.8 * len(data)):int(0.9 * len(data))]])

    split.insert_many([{
        'conj': d['goal'],
        'stmt': d['premise'],
        'y': d['y'],
        'split': 'test'
    }
        for d in data[int(0.9 * len(data)):]])


    # todo
    def sexpression_to_polish(sexpression_text):
        sexpression = SExpressionGraph()
        sexpression.add_sexp(sexpression_text)
        out = []

        def process_node(node):
            if len(sexpression.get_children(node)) == 0:
                out.append(node)
            for i, child in enumerate(sexpression.get_children(node)):
                # todo look at lean sexp, append e.g. special character for each child
                if i == 0:
                    out.append(sexpression.to_text(child))
                    continue
                process_node(sexpression.to_text(child))

        process_node(sexpression.to_text(sexpression.roots()[0]))
        return out


    # compute sexpression data for all expressions and add to database
    expr_col = db['expression_graphs']

    logging.info('Computing and adding sexpression graph information to database..')
    for expr in tqdm(all_exprs):
        g = sexpression_to_graph(expr)

        expr_col.insert_one({
            '_id': expr,
            'data': {
                'tokens': g['tokens'],
                'edge_index': g['edge_index'],
                'edge_attr': g['edge_attr']
            }
        })

    # compute vocab

    tok_set = set()

    for g in tqdm(expr_col.find()):
        tok_set = tok_set | set(g['data']['tokens'])

    vocab = {}

    i = 1
    for tok in tok_set:
        vocab[tok] = i
        i += 1

    # len(vocab)

    db['vocab'].insert_many([{'_id': k, 'index': v} for k, v in vocab.items()])
