import warnings

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from data.utils.dataset import H5DataModule

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from data.hol4.mongo_to_torch import HOL4DataModule, HOL4DataModuleGraph, HOL4SequenceModule
from lightning.pytorch.loggers import WandbLogger
from models.get_model import get_model
from models.gnn.formula_net.formula_net import BinaryClassifier
import torch
from collections import namedtuple
from data.mizar.mizar_data_module import MizarDataModule

data_tuple = namedtuple('data_tuple', 'graph_dict, expr_dict, train_data, val_data, test_data')

def binary_loss(preds, targets):
    return -1. * torch.mean(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


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
        try:
            preds = self(goal, premise)
        except Exception as e:
            print (f"Error in forward: {e}")
            return
        loss = binary_loss(preds, y)
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('acc', summary='max')

        goal, premise, y = batch
        try:
            preds = self(goal, premise)
            preds = (preds > 0.5)
            acc = torch.sum(preds == y) / y.size(0)
            self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)
        except Exception as e:
            print(f"Error in val forward {e}")
        return

    def test_step(self, batch, batch_idx):
        goal, premise, y = batch
        preds = self(goal, premise)
        preds = (preds > 0.5)
        acc = torch.sum(preds == y) / y.size(0)
        self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            print (f"Error in backward: {e}")


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
    if data_config['source'] == 'hol4':
        return HOL4DataModule(dir=data_config['data_dir'])
    if data_config['source'] == 'hol4_graph':
        return HOL4DataModuleGraph(dir=data_config['data_dir'])
    if data_config['source'] == 'hol4_sequence':
        return HOL4SequenceModule(dir=data_config['data_dir'])
    if data_config['source'] == 'mizar':
        return MizarDataModule(dir=data_config['data_dir'])
    else:
        raise NotImplementedError



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
                             notes=self.config['notes'],
                             # offline=True,
                             )

        callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="acc", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{acc}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              save_weights_only=True,
                                              dirpath=self.exp_config['checkpoint_dir'])


        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=self.exp_config['epochs'],
            val_check_interval=self.exp_config['val_frequency'],
            limit_val_batches=self.exp_config['val_size'] // self.exp_config['batch_size'],
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=500,
            # accelerator='cpu',
            devices=self.exp_config['device'],
            enable_checkpointing=True,
            callbacks=callbacks,
            )

        trainer.fit(model=experiment, datamodule=data_module)
        logger.experiment.finish()