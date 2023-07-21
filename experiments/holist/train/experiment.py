from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from experiments.holist.train import torch_training_module
from holist.data.sequence_datamodule import HOListSequenceModule
from holist.data.graph_datamodule import HOListGraphModule
from holist.models.get_model import get_model
from holist.models.tactic_predictor import TacticPrecdictor, CombinerNetwork


class ExperimentRunner:
    def __init__(self, exp_config, embedding_model_config):
        if exp_config['data_type'] == 'sequence':
            self.data_module = HOListSequenceModule(dir=exp_config['data_dir'], batch_size=exp_config['batch_size'])
        else:
            self.data_module = HOListGraphModule(dir=exp_config['data_dir'], batch_size=exp_config['batch_size'])

        self.experiment = torch_training_module.HOListTraining_(embedding_model_goal=get_model(embedding_model_config),
                                                                embedding_model_premise=get_model(embedding_model_config),
                                                                tac_model=TacticPrecdictor(
                                                                    embedding_dim=exp_config['final_embed_dim'],
                                                                    num_tactics=exp_config['num_tactics']),
                                                                combiner_model=CombinerNetwork(
                                                                    embedding_dim=exp_config['final_embed_dim'],
                                                                    num_tactics=exp_config['num_tactics'],
                                                                    tac_embed_dim=exp_config['tac_embed_dim']),
                                                                lr=exp_config['learning_rate'],
                                                                batch_size=exp_config['batch_size'])

        self.logger = WandbLogger(project=exp_config['project'],
                                  name=exp_config['name'],
                                  config={'model_config': embedding_model_config, 'exp_config': exp_config},
                                  )

        self.callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              save_weights_only=True,
                                              dirpath=exp_config['checkpoint_dir'])

        self.callbacks.append(checkpoint_callback)


        batch_scale = (16 // exp_config['batch_size'])
        self.trainer = pl.Trainer(enable_progress_bar=True,
                                  log_every_n_steps=50 * batch_scale,
                                  max_epochs=exp_config['epochs'],
                                  devices=exp_config['device'],
                                  val_check_interval=exp_config['val_frequency'] * batch_scale,
                                  limit_val_batches=exp_config['val_size'] * batch_scale,
                                  logger=self.logger,
                                  callbacks=self.callbacks)

    def run(self):
        self.trainer.fit(model=self.experiment, datamodule=self.data_module)
