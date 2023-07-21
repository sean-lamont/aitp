from experiments.holist.utilities.sexpression_graphs import SExpressionGraph
from experiments.holist.utilities import sexpression_graphs
from typing import Text
import logging


def sexpression_to_graph(sexpression_txt: Text):
    sexpression = SExpressionGraph(sexpression_txt)

    edges = []
    node_to_tok = {}

    def process_sexpression_graph(node, depth):
        node_id = sexpression_graphs.to_node_id(node)

        if len(sexpression.get_children(node)) == 0:
            if node_id not in node_to_tok:
                node_to_tok[node_id] = node
            assert sexpression.is_leaf_node(node_id)

        for i, child in enumerate(sexpression.get_children(node)):
            if i == 0:
                node_to_tok[node_id] = sexpression.to_text(child)
                continue

            edges.append((node_id, child, i))
            process_sexpression_graph(sexpression.to_text(child), depth + 1)

    if len(sexpression.roots()) > 1:
        logging.warning(
            f"Multiple roots for {sexpression_txt}: {[sexpression.to_text(sexpression.roots()[i]) for i in sexpression.roots()]}")

    process_sexpression_graph(sexpression.to_text(sexpression.roots()[0]), 0)

    edges = set(edges)
    senders = [a[0] for a in edges]
    receivers = [a[1] for a in edges]
    edge_attr = [a[2] for a in edges]

    all_nodes = list(set(senders + receivers))
    senders = [all_nodes.index(i) for i in senders]
    receivers = [all_nodes.index(i) for i in receivers]

    node_to_tok_ = {}
    for k, v in node_to_tok.items():
        node_to_tok_[all_nodes.index(k)] = v

    assert len(node_to_tok_) == len(all_nodes)

    tok_list = [0 for _ in range(len(all_nodes))]

    for k, v in node_to_tok_.items():
        tok_list[k] = v

    # todo add attention_edge and depth here as an option

    return {'tokens': tok_list, 'edge_index': [senders, receivers], 'edge_attr': edge_attr}
