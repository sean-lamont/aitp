from data.utils.dataset import H5DataModule
import warnings
warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from models import gnn_edge_labels
from models.get_model import get_model
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
        self.log("acc", acc, batch_size=self.batch_size)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

class PremiseSelectionExperiment:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model_config']
        self.data_config = config['data_config']
        self.exp_config = config['exp_config']

# todo combine into one experiment class, which generates experiment and data from config i.e. get_exp, get_data
# todo analogous to get_model
'''
Premise selection experiment with separate encoders for goal and premise
'''
class SeparateEncoderPremiseSelection(PremiseSelectionExperiment):
    def __int__(self, config):
        super(self, config)
        self.save_hyperparameters()

    def run_lightning(self):
        torch.set_float32_matmul_precision('high')

        ps = PremiseSelection(get_model(self.model_config),
                              get_model(self.model_config),
                              gnn_edge_labels.F_c_module_(self.model_config['embedding_dim'] * 2),
                              lr=self.exp_config['learning_rate'],
                              batch_size=self.exp_config['batch_size'])

        data_module = H5DataModule(config=self.data_config)

        logger = WandbLogger(project=self.config['project'],
                             name=self.config['name'],
                             config=self.config,
                             offline=False)

        trainer = pl.Trainer(
                            max_epochs=self.exp_config['epochs'],
                             val_check_interval=self.exp_config['val_frequency'],
                             limit_val_batches=self.exp_config['val_size'] // self.exp_config['batch_size'],
                             logger=logger,
                             enable_progress_bar=True,
                             log_every_n_steps=500,
                             # accelerator='gpu',
                             # devices=2,
                            # todo figure out why, e.g. https://github.com/Lightning-AI/lightning/issues/11242
                             # hack to fix ddp hanging error..
                             limit_train_batches=28000,
                             # profiler='pytorch',
                             enable_checkpointing=True)

        # trainer.logger.watch(ps, log_freq=500)
        trainer.fit(model=ps, datamodule=data_module)