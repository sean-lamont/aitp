from experiments.hol4.rl.lightning_rl.agent_utils import *
from experiments.hol4.rl.lightning_rl.rl_data_module import *
import torch.optim
from data.hol4.ast_def import graph_to_torch_labelled
import pickle
from data.hol4 import ast_def
from environments.hol4.new_env import *
import warnings

warnings.filterwarnings('ignore')
from experiments.hol4.rl.lightning_rl.rl_experiment import RLExperiment


####################################################################################################
# RELATION
####################################################################################################
def run_rl_relation():
    premise_db = {}

    print("Generating premise graph db...")
    for i, t in enumerate(compat_db):
        premise_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc)

    VOCAB_SIZE = 1004
    EMBEDDING_DIM = 256

    model_config = {'model_type': 'transformer_relation',
                    'vocab_size': VOCAB_SIZE,
                    'embedding_dim': EMBEDDING_DIM}

    relation_config = default_config
    relation_config['name'] = 'Relation 50 Step'
    relation_config['model_config'] = model_config
    relation_config[
        'pretrain_ckpt'] = "/home/sean/Documents/phd/repo/aitp/sat/hol4/supervised/model_checkpoints/epoch=5-step=41059.ckpt"
    relation_config['exp_type'] = 'relation'
    relation_config['data_type'] = 'relation'
    relation_config['vocab_size'] = VOCAB_SIZE
    relation_config['notes'] = 'relation_50_step/'
    relation_config['graph_db'] = premise_db
    relation_config['embedding_dim'] = EMBEDDING_DIM

    # relation_config['resume'] = True
    # relation_config['pretrain'] = False
    # relation_50_id = '3u17ha2u'

    experiment = RLExperiment(relation_config)
    experiment.run()


####################################################################################################
# TRANSFORMER
####################################################################################################
def run_rl_transformer():
    print("Generating Premise DB..")
    premise_db = {}

    def to_sequence(data, vocab):
        data = data.split(" ")
        data = torch.LongTensor([vocab[c] for c in data])
        return data

    vocab = {}
    i = 1
    for k in compat_db.keys():
        tmp = k.split(" ")
        for t in tmp:
            if t not in vocab:
                vocab[t] = i
                i += 1

    for i, t in enumerate(compat_db):
        premise_db[t] = to_sequence(t, vocab)

    premise_db = premise_db, vocab

    VOCAB_SIZE = 1300
    EMBEDDING_DIM = 256

    model_config = {
        "model_type": "transformer",
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "dim_feedforward": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.0
    }

    transformer_config = default_config
    transformer_config['name'] = 'Transformer 50 Step'
    transformer_config['model_config'] = model_config
    transformer_config[
        'pretrain_ckpt'] = "/home/sean/Documents/phd/repo/aitp/experiments/hol4/supervised/model_checkpoints/transformer_90_04.ckpt"
    transformer_config['exp_type'] = 'sequence'
    transformer_config['data_type'] = 'sequence'
    transformer_config['vocab_size'] = VOCAB_SIZE
    transformer_config['notes'] = 'transformer_50_step/'
    transformer_config['graph_db'] = premise_db
    transformer_config['embedding_dim'] = EMBEDDING_DIM

    # transformer_config['resume'] = True
    # transformer_config['pretrain'] = False
    # transformer_50_id = '6wwliaih'

    experiment = RLExperiment(transformer_config)
    experiment.run()


####################################################################################################
# GNN
####################################################################################################

def run_rl_gnn():
    premise_db = {}

    print("Generating premise graph db...")
    for i, t in enumerate(compat_db):
        premise_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc)

    VOCAB_SIZE = 1004
    EMBEDDING_DIM = 256

    model_config = {
        "model_type": "formula-net-edges",
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "gnn_layers": 3,
        "batch_norm": False
    }

    gnn_config = default_config
    gnn_config['name'] = 'GNN 50 Step'
    gnn_config['model_config'] = model_config
    gnn_config[
        'pretrain_ckpt'] = "/home/sean/Documents/phd/repo/aitp/experiments/hol4/supervised/model_checkpoints/formula_net_best_91.ckpt"
    gnn_config['exp_type'] = 'gnn'
    gnn_config['data_type'] = 'graph'
    gnn_config['vocab_size'] = VOCAB_SIZE
    gnn_config['notes'] = 'gnn_50_step_test/'
    gnn_config['graph_db'] = premise_db
    gnn_config['embedding_dim'] = EMBEDDING_DIM

    experiment = RLExperiment(gnn_config)
    experiment.run()

    # gnn_config['resume'] = True
    # gnn_config['pretrain'] = False
    # gnn_config['resume_id'] = 'qf5w6qk0'


if __name__ == '__main__':
    with open("data/hol4/data/valid_goals_shuffled.pk", "rb") as f:
        valid_goals = pickle.load(f)

    train_goals = valid_goals[:int(0.8 * len(valid_goals))]
    test_goals = valid_goals[int(0.8 * len(valid_goals)):]

    with open("data/hol4/data/graph_token_encoder.pk", "rb") as f:
        token_enc = pickle.load(f)

    encoded_graph_db = []
    with open('data/hol4/data/adjusted_db.json') as f:
        compat_db = json.load(f)

    reverse_database = {(value[0], value[1]): key for key, value in compat_db.items()}

    MORE_TACTICS = True
    if not MORE_TACTICS:
        thms_tactic = ["simp", "fs", "metis_tac"]
        thm_tactic = ["irule"]
        term_tactic = ["Induct_on"]
        no_arg_tactic = ["strip_tac"]
    else:
        thms_tactic = ["simp", "fs", "metis_tac", "rw"]
        thm_tactic = ["irule", "drule"]
        term_tactic = ["Induct_on"]
        no_arg_tactic = ["strip_tac", "EQ_TAC"]

    tactics = {
        'thms_tactic': thms_tactic,
        'thm_tactic': thm_tactic,
        'term_tactic': term_tactic,
        'no_arg_tactic': no_arg_tactic,
        'tactic_pool': thms_tactic + thm_tactic + term_tactic + no_arg_tactic
    }

    dir = '/home/sean/Documents/phd/repo/aitp/experiments/hol4/rl/lightning_rl/experiments/'

    default_config = {
        'model_config': None,
        'project': 'RL Test',
        'name': None,  # 'Relation 50 Step',
        'tactics': tactics,
        'pretrain_ckpt': None,
        'exp_type': None,
        'vocab_size': None,
        'embedding_dim': None,
        'resume': False,
        'resume_id': None,
        'pretrain': True,
        'gnn_layers': 3,
        'notes': None,  # 'x_50_step/',
        'dir': dir,
        'max_steps': 50,
        'gamma': 0.99,
        'lr': 5e-5,
        'arg_len': 5,
        'data_type': None,  # 'graph' 'relation' 'sequence'
        'train_goals': train_goals,
        'test_goals': test_goals,
        'token_enc': token_enc,
        'graph_db': None,
        'device': [1],
        'val_freq': 5,
        'reverse_database': reverse_database
    }

    torch.set_float32_matmul_precision('high')

    run_rl_gnn()
