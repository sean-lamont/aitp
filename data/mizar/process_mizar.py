import glob
import random
import pickle

from ast_def_mizar import goal_to_graph, graph_to_dict
from tqdm import tqdm

if __name__ == '__main__':

    files = glob.glob('nnhpdata/*')

    expression_dict = {}
    mizar_labels = []

    for file in tqdm(files):
        pos_thms = []
        neg_thms = []

        with open(file) as f:
            lines = f.readlines()

        assert lines[0][0] == 'C'

        for line in lines:
            if line[0] == 'C':
                conj = line[1:].strip("\n")
                # if conj not in expression_dict:
                #     expression_dict[conj] = graph_to_dict(goal_to_graph(conj))
            elif line[0] == '-':
                neg_thm = line[1:].strip("\n")
                neg_thms.append(neg_thm)
                # if neg_thm not in expression_dict:
                #     expression_dict[neg_thm] = graph_to_dict(goal_to_graph(neg_thm))
                # mizar_labels.append((conj, neg_thm, 0))
            elif line[0] == '+':
                pos_thm = line[1:].strip("\n")
                pos_thms.append(pos_thm)
                # if pos_thm not in expression_dict:
                #     expression_dict[pos_thm] = graph_to_dict(goal_to_graph(pos_thm))
                # mizar_labels.append((conj, pos_thm, 1))
            else:
                raise Exception("Not valid")

        mizar_labels.append((conj, pos_thms, neg_thms))


    random.shuffle(mizar_labels)

    train_data = mizar_labels[:int(0.8 * len(mizar_labels))]
    val_data = mizar_labels[
                    int(0.8 * len(mizar_labels)):int(0.9 * len(mizar_labels))]

    test_data = mizar_labels[int(0.9 * len(mizar_labels)):]

    train_pairs = []
    for conj, pos_thms, neg_thms in train_data:
        for pos_thm in pos_thms:
            train_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            train_pairs.append((conj, neg_thm, 0))

    val_pairs = []
    for conj, pos_thms, neg_thms in val_data:
        for pos_thm in pos_thms:
            val_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            val_pairs.append((conj, neg_thm, 0))

    test_pairs = []
    for conj, pos_thms, neg_thms in test_data:
        for pos_thm in pos_thms:
            test_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            test_pairs.append((conj, neg_thm, 0))



    # vocab = {}
    # idx = 0
    # for i, k in enumerate(expression_dict.keys()):
    #     polished_goal = [c for c in k.split(" ") if c != '' and c != '\n']
    #     for tok in polished_goal:
    #         if tok not in vocab:
    #             # reserve 0 for padding idx
    #             vocab[tok] = idx + 1
    #             idx += 1
    #
    # vocab['VAR'] = len(vocab)
    # vocab['VARFUNC'] = len(vocab)
    #


    # with open("mizar_data_.pk", "wb") as f:
    #     pickle.dump({'expr_dict': expression_dict, 'mizar_labels': mizar_labels, 'vocab': vocab}, f)

    with open("mizar_data_.pk", "wb") as f:
        pickle.dump({'train_data': train_pairs, 'val_data': val_pairs, 'test_data': test_pairs}, f)

