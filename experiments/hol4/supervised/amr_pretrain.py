import wandb
import optuna
import cProfile
from data.utils.pretrain import SeparateEncoderPremiseSelection
from lightning import LightningApp

# HOL4 vocab
VOCAB_SIZE = 1000

# HOLStep vocab
VOCAB_SIZE = 1909 + 4


sat_config = {
    "model_type": "graph_benchmarks",
    "num_edge_features":  200,
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
    "dropout": 0.,
    "gnn_layers": 4,
    "directed_attention": False,
}

transformer_config = {
    "model_type": "transformer",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.0
}

relation_config = {
    "model_type": "transformer_relation",
    "vocab_size": VOCAB_SIZE,
    # "vocab_size": VOCAB_SIZE + 1,
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.0
}

amr_config = {
    "model_type": "amr",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "dim_feedforward": 512,
    "num_heads": 4,
    "num_layers": 4,
    "in_embed": True,
    "abs_pe": False,
    "abs_pe_dim": 2,
    "use_edge_attr": True,
    "device": "cuda:0",
    "dropout": 0.,
}

exp_config = {
    "experiment": "premise_selection",
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "model_save": False,
    "val_size": 4096,
    "logging": False,
    "checkpoint_dir": "/home/sean/Documents/phd/repo/aitp/sat/hol4/supervised/model_checkpoints",
    "device": "cuda:0",
    "max_errors": 1000,
    "val_frequency": 2048
}

formula_net_config = {
    "model_type": "formula-net-edges",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "gnn_layers": 3,
}

digae_config = {
    "model_type": "digae",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256
}


h5_data_config = {"source": "h5", "data_dir": "/home/sean/Documents/phd/repo/aitp/data/utils/holstep_full"}

relation_att_exp = SeparateEncoderPremiseSelection(config={"model_config": relation_config,
                                                           "exp_config": exp_config,
                                                           "data_config": h5_data_config,
                                                           "project": "test_project",
                                                           "name": "relation attention large"})

# todo with original sequence for positional encoding
transformer_experiment = SeparateEncoderPremiseSelection(config={"model_config": transformer_config,
                                                                 "exp_config": exp_config,
                                                                 "data_config": h5_data_config,
                                                                 "project": "test_project",
                                                                "name": "transformer"})

sat_exp = SeparateEncoderPremiseSelection(config={"model_config": sat_config,
                                                           "exp_config": exp_config,
                                                           "data_config": h5_data_config,
                                                           "project": "test_project",
                                                           "name": "graph_benchmarks"})


formula_net_exp = SeparateEncoderPremiseSelection(config={"model_config": formula_net_config,
                                                           "exp_config": exp_config,
                                                           "data_config": h5_data_config,
                                                           "project": "test_project",
                                                           "name": "formula_net_batch_norm_edges"})

digae_exp = SeparateEncoderPremiseSelection(config={"model_config": digae_config,
                                                          "exp_config": exp_config,
                                                          "data_config": h5_data_config,
                                                          "project": "test_project",
                                                          "name": "digae_large"})

amr_exp = SeparateEncoderPremiseSelection(config={"model_config": amr_config,
                                                           "exp_config": exp_config,
                                                           "data_config": h5_data_config,
                                                           "project": "test_project",
                                                           "name": "amr"})


# amr_exp.run_lightning()
# sat_exp.run_lightning()
relation_att_exp.run_lightning()
# formula_net_exp.run_lightning()
# transformer_experiment.run_lightning()
# digae_exp.run_lightning()




# study =  optuna.create_study(direction='maximize')
# study.optimize(relation_att_exp.objective, n_trials=2,timeout=100)
# study.optimize(relation_att_exp.objective,timeout=100)



# cProfile.run('transformer_experiment.run_dual_encoders()', sort ='cumtime')
# cProfile.run('transformer_experiment.run_lightning()', sort='cumtime')

# transformer_experiment = MaskPretrain(config = {"model_config": relation_config,
#                                                                    "exp_config": exp_config,
#                                                                    "data_config": mask_config})
# transformer_experiment.run_mask_experiment()


# run_dual_encoders(config = {"model_config": sat_config, "exp_config": exp_config, "data_config": data_config})

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
# def main():
#     wandb.init(
#         project="hol4_premise_selection",
#
#         name="Directed Attention Sweep Separate Encoder",

#         # track model and experiment configurations
#         config={
#             "exp_config": exp_config,
#             "model_config": amr_config,
#             "data_config": data_config
#         }
#     )
#
#     wandb.define_metric("acc", summary="max")
#
#     run_dual_encoders(wandb.config)
#
#     return
#
