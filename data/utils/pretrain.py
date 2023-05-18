from data.utils.dataset import H5DataModule
import wandb
import os
import warnings

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from models.get_model import get_model
from models.gnn.formula_net.formula_net import BinaryClassifier
import torch


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


# todo init takes in config file with model and exp details

class PremiseSelection(pl.LightningModule):
    def __init__(self,
                 embedding_model_goal,
                 embedding_model_premise,
                 classifier,
                 batch_size=32,
                 lr=1e-4):
        super().__init__()

        self.embedding_model_goal = embedding_model_goal
        self.embedding_model_premise = embedding_model_premise
        self.classifier = classifier
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = batch_size

        self.save_hyperparameters()

    def forward(self, goal, premise):
        embedding_goal = self.embedding_model_goal(goal)
        embedding_premise = self.embedding_model_premise(premise)
        preds = self.classifier(torch.cat([embedding_goal, embedding_premise], dim=1))
        preds = torch.clip(preds, self.eps, 1 - self.eps)
        return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        goal, premise, y = batch
        preds = self(goal, premise)
        loss = binary_loss(preds, y)
        # loss = torch.nn.functional.cross_entropy(preds, y)
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        goal, premise, y = batch
        preds = self(goal, premise)
        preds = (preds > 0.5)
        acc = torch.sum(preds == y) / y.size(0)
        self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def get_experiment(exp_config, model_config):
    if exp_config['experiment'] == 'premise_selection':
        return PremiseSelection(get_model(model_config),
                                get_model(model_config),
                                BinaryClassifier(model_config['embedding_dim'] * 2),
                                lr=exp_config['learning_rate'],
                                batch_size=exp_config['batch_size'])
    else:
        raise NotImplementedError

    # todo combine into one experiment class, which generates experiment and data from config i.e. get_exp, get_data


# todo MongoDB
def get_data(data_config):
    if data_config['source'] == 'h5':
        return H5DataModule(config=data_config)
    else:
        raise NotImplementedError



# todo analogous to get_model

'''
Premise selection experiment with separate encoders for goal and premise
'''


class SeparateEncoderPremiseSelection:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model_config']
        self.data_config = config['data_config']
        self.exp_config = config['exp_config']

    def run_lightning(self):
        torch.set_float32_matmul_precision('high')

        experiment = get_experiment(self.exp_config, self.model_config)

        data_module = get_data(self.data_config)

        logger = WandbLogger(project=self.config['project'],
                             name=self.config['name'],
                             config=self.config,
                             offline=True)

        trainer = pl.Trainer(
            max_epochs=self.exp_config['epochs'],
            val_check_interval=self.exp_config['val_frequency'],
            limit_val_batches=self.exp_config['val_size'] // self.exp_config['batch_size'],
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=500,
            # accelerator='gpu',
            # devices=1,
            # strategy='ddp_find_unused_parameters_true',
            # todo figure out why, e.g. https://github.com/Lightning-AI/lightning/issues/11242
            # hack to fix ddp hanging error..
            limit_train_batches=28000,
            # profiler='pytorch',
            enable_checkpointing=True)

        trainer.fit(model=experiment, datamodule=data_module)
#
# def objective(self, trial):
#     torch.set_float32_matmul_precision('high')
#
#     self.exp_config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         os.path.join(self.exp_config['checkpoint_dir'], "trial_{}".format(trial.number)), monitor="acc"
#     )
#
#
#     ps = PremiseSelection(get_model(self.model_config),
#                           get_model(self.model_config),
#                           gnn_edge_labels.F_c_module_(self.model_config['embedding_dim'] * 2),
#                           lr=self.exp_config['learning_rate'],
#                           batch_size=self.exp_config['batch_size'])
#
#     data_module = H5DataModule(config=self.data_config)
#
#     logger = WandbLogger(project=self.config['project'],
#                          name=self.config['name'], config=self.exp_config)
#
#     trainer = pl.Trainer(
#         max_epochs=self.exp_config['epochs'],
#         val_check_interval=self.exp_config['val_frequency'],
#         limit_val_batches=self.exp_config['val_size'] // self.exp_config['batch_size'],
#         logger=logger,
#         callbacks=[checkpoint_callback],
#         max_steps=100,
#         enable_progress_bar=True,
#         log_every_n_steps=500,
#         accelerator='gpu',
#         devices=1,
#         # limit_train_batches=28000,
#         # profiler='pytorch',
#         enable_checkpointing=True)
#
#     trainer.fit(model=ps, datamodule=data_module)
#
#     logger.experiment.finish()
#
#     # early_stopping_callback.check_pruned()
#
#     return trainer.callback_metrics["acc"].item()
#


#
#
# model_config = {
#     "model_type": "transformer_relation",
#     "vocab_size": 1909,
#     # "vocab_size": VOCAB_SIZE + 1,
#     "embedding_dim": 128,
#     "dim_feedforward": 512,
#     "num_heads": 8,
#     "num_layers": 4,
#     "dropout": 0.2
# }
#
# exp_config = {
#     "learning_rate": 1e-4,
#     "epochs": 20,
#     "weight_decay": 1e-6,
#     "batch_size": 32,
#     "model_save": False,
#     "val_size": 4096,
#     "logging": False,
#     "checkpoint_dir": "/home/sean/Documents/phd/repo/aitp/sat/hol4/supervised/model_checkpoints",
#     "device": "cuda:0",
#     # "device": "cpu",
#     "max_errors": 1000,
#     "val_frequency": 100
# }
#
# data_config = {"data_dir": "/home/sean/Documents/phd/repo/aitp/data/utils/processed_data"}
# #
#
# def run_exp(config, trial_num):
#     model_config = config['model_config']
#     exp_config = config['exp_config']
#     data_config = config['data_config']
#
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         os.path.join(exp_config['checkpoint_dir'], "trial_{}".format(trial_num)), monitor="acc"
#     )
#
#     ps = PremiseSelection(get_model(model_config),
#                           get_model(model_config),
#                           gnn_edge_labels.F_c_module_(model_config['embedding_dim'] * 2),
#                           lr=exp_config['learning_rate'],
#                           batch_size=exp_config['batch_size'])
#
#     data_module = H5DataModule(config=data_config)
#
#     logger = WandbLogger(project='test_project',
#                          name='test', config=exp_config)
#
#     trainer = pl.Trainer(
#         max_epochs=exp_config['epochs'],
#         val_check_interval=exp_config['val_frequency'],
#         limit_val_batches=exp_config['val_size'] // exp_config['batch_size'],
#         logger=logger,
#         callbacks=[checkpoint_callback],
#         # max_steps=100,
#         enable_progress_bar=True,
#         log_every_n_steps=500,
#         accelerator='gpu',
#         devices=1,
#         # todo figure out why, e.g. https://github.com/Lightning-AI/lightning/issues/11242
#         # hack to fix ddp hanging error..
#         # limit_train_batches=28000,
#         # profiler='pytorch',
#         enable_checkpointing=True)
#
#     trainer.fit(model=ps, datamodule=data_module)
#     logger.experiment.finish()
#
#     # early_stopping_callback.check_pruned()
#
#     return trainer.callback_metrics["acc"].item()
#
# def objective(trial):
#     torch.set_float32_matmul_precision('high')
#
#     exp_config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#     exp_config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#
#     return run_exp(config= {"model_config": model_config, "exp_config": exp_config, "data_config":data_config}, trial_num=trial.number)
