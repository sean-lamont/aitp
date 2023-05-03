import wandb
from data.utils.pretrain import run_dual_encoders

# HOL4 vocab
VOCAB_SIZE = 1000

# HOLStep vocab
VOCAB_SIZE = 1909

# AMR data HOL4
data_config = {
    "source_config":{
    "data_source": "MongoDB",
    "dbname": "hol4_tactic_zero",
    "graph_collection_name": "expression_graph_data",
    "split_name": "pretrain_data"
    },
    'data_options' : ['softmax_idx', 'edge_attr', 'edge_index']
}

# AMR data HOLStep
data_config = {
    "source_config":{
        "data_source": "MongoDB",
        "dbname": "hol_step",
        "graph_collection_name": "expression_graphs",
        "split_name": "train_val_test_data"
    },
    'data_options' : ['softmax_idx', 'edge_attr', 'edge_index']
}

# SAT data HOL4
# data_config = {
#     "source_config":{
#         "data_source": "MongoDB",
#         "dbname": "hol4_tactic_zero",
#         "graph_collection_name": "expression_graph_data",
#         "split_name": "pretrain_data"
#     },
#     'data_options' : ['edge_attr', 'edge_index', 'attention_edge_index']
# }

# SAT data HOLStep
# data_config = {
#     "source_config":{
#         "data_source": "MongoDB",
#         "dbname": "hol_step",
#         "graph_collection_name": "expression_graphs",
#         "split_name": "train_val_test_data"
#     },
#     'data_options' : ['attention_edge_index', 'edge_attr', 'edge_index']
# }

sat_config = {
    "model_type": "sat",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "dim_feedforward": 256,
    "num_heads": 8,
    "num_layers": 4,
    "in_embed": True,
    "se": "pna",
    "abs_pe": False,
    "abs_pe_dim": 256,
    "use_edge_attr": True,
    "dropout": 0.2,
    "gnn_layers": 4,
    "directed_attention": False,
}


amr_config = {
    "model_type": "amr",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "in_embed": True,
    "abs_pe": False,
    "abs_pe_dim":2,
    "use_edge_attr": True,
    "device": "cuda:0",
    "dropout": 0.2,
}

exp_config = {
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "model_save": False,
    "val_size": 2048,
    "logging": False,
    "model_dir": "/home/sean/Documents/phd/aitp/experiments/hol4/supervised/model_checkpoints",
    "device": "cuda:0",
    # "device": "cpu",
    "max_errors": 1000,
    "val_frequency": 1000
}

formula_net_config = {
    "model_type": "formula-net",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "gnn_layers": 4,
}


def main():
    wandb.init(
        project="hol4_premise_selection",

        name="Directed Attention Sweep Separate Encoder",
        # track model and experiment configurations
        config={
            "exp_config": exp_config,
            "model_config": amr_config,
            "data_config": data_config
        }
    )

    wandb.define_metric("acc", summary="max")

    run_dual_encoders(wandb.config)

    return


import cProfile
# cProfile.run('run_dual_encoders(config = {"model_config": amr, "exp_config": exp_config})', sort='cumtime')

import cProfile
cProfile.run('run_dual_encoders(config = {"model_config": amr_config, "exp_config": exp_config, "data_config": data_config})', sort='cumtime')

# sweep_configuration = {
#     "method": "bayes",
#     "metric": {'goal': 'maximize', 'name': 'acc'},
#     "parameters": {
#         "model_config" : {
#             "parameters": {
#                 "model_type": {"values":["sat"]},
#                 "vocab_size": {"values":[len(tokens)]},
#                 "embedding_dim": {"values":[128]},
#                 "in_embed": {"values":[False]},
#                 "abs_pe": {"values":[True, False]},
#                 "abs_pe_dim": {"values":[128]},
#                 "use_edge_attr": {"values":[True, False]},
#                 "dim_feedforward": {"values": [256]},
#                 "num_heads": {"values": [8]},
#                 "num_layers": {"values": [4]},
#                 "se": {"values": ['pna']},
#                 "dropout": {"values": [0.2]},
#                 "gnn_layers": {"values": [0,4]},
#                 "directed_attention": {"values": [True,False]}
#             }
#         }
#     }
# }
#
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='hol4_premise_selection')
# #
# wandb.agent(sweep_id,function=main)
#
