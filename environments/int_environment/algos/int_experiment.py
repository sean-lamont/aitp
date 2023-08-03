import json
import logging
import os
import pickle
from datetime import datetime
from time import time

import lightning.pytorch as pl
import torch
import torch.utils.data as data_handler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from environments.int_environment.algos.eval import eval_agent
from environments.int_environment.algos.lib.arguments import get_args
from environments.int_environment.algos.lib.obs import nodename2index, thm2index, batch_process

# todo use get_model with config
from environments.int_environment.data_generation.generate_problems import generate_multiple_problems
from environments.int_environment.data_generation.utils import Dataset
from experiments.int_pyrallis import ThmProvingConfig
from experiments.pyrallis_configs import ExperimentConfig, LoggingConfig, OptimiserConfig

timestamp = str(datetime.fromtimestamp(time())).replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")

args = get_args()
args.use_gpu = args.cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.use_gpu else "cpu")

if args.num_order_or_combo < 0:
    args.num_order_or_combo = None

os.makedirs(os.path.join(args.dump, str(timestamp)))

def load_data(data_dir, mode="train"):
    file_name = os.path.join(data_dir, '{}.pkl'.format(mode))
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


dgl = (args.obs_mode == "dgl")
bow = args.bag_of_words
print(args.transform_gt)
if dgl and (not bow):
    from environments.int_environment.algos.model.old.thm_model_dgl import ThmNet
elif bow and (not dgl):
    from environments.int_environment.algos.model.thm_model import ThmNet
elif (not bow) and (not dgl):
    from environments.int_environment.algos.model.thm_model import ThmNet
else:
    raise AssertionError


def load_all_data(train_dirs, test_dirs):
    train_dataset = Dataset([])
    val_dataset = Dataset([])
    train_first_dataset = Dataset([])
    val_first_dataset = Dataset([])
    for train_dir in train_dirs:
        train_ds = load_data(train_dir, mode="train")
        train_dataset.merge(train_ds)
        val_ds = load_data(train_dir, mode="val")
        val_dataset.merge(val_ds)
        train_first_ds = load_data(train_dir, mode="train_first")
        train_first_dataset.merge(train_first_ds)
        val_first_ds = load_data(train_dir, mode="val_first")
        val_first_dataset.merge(val_first_ds)

    test_dataset = Dataset([])
    test_first_dataset = Dataset([])
    for test_dir in test_dirs:
        test_ds = load_data(test_dir, mode="test")
        test_dataset.merge(test_ds)
        test_first_ds = load_data(test_dir, mode="test_first")
        test_first_dataset.merge(test_first_ds)

    return {
        "train": train_dataset,
        "train_first": train_first_dataset,
        "val": val_dataset,
        "val_first": val_first_dataset,
        "test": test_dataset,
        "test_first": test_first_dataset
    }


class INTDataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        # self.config = config

        super().__init__()
        if not args.online:
            train_dirs = [os.path.join(args.path_to_data, train_dir) for train_dir in args.train_sets]
            test_dirs = [os.path.join(args.path_to_data, test_dir) for test_dir in args.test_sets]
            all_data = load_all_data(train_dirs, test_dirs)

            (self.train_dataset, self.val_dataset, self.eval_dataset, self.train_first_dataset,
             self.val_first_dataset, self.eval_first_dataset) = (all_data["train"], all_data["val"], all_data["test"],
                                                                 all_data["train_first"], all_data["val_first"],
                                                                 all_data["test_first"])
        else:
            if args.online_order_generation:
                self.kl_dict = json.load(open(os.path.join(args.combo_path, "combinations.json"), "r"))
            else:
                self.kl_dict = json.load(open(os.path.join(args.combo_path, "orders.json"), "r"))

            self.val_dataset = Dataset([])
            self.eval_dataset = Dataset([])
            self.eval_first_dataset = Dataset([])

            for kl in args.test_sets:
                k = kl.split("_")[0][-1]
                l = int(kl[-1])

                data_path = os.path.join(args.combo_path,
                                         "test_first_dataset_prob{}_k{}l{}_oog{}_nooc{}_degree{}.pkl".format(
                                             args.num_probs, k, l,
                                             args.online_order_generation, args.num_order_or_combo,
                                             args.degree)
                                         )

                if os.path.isfile(data_path):
                    with pickle.load(open(data_path, "rb")) as existent_dataset:
                        self.eval_first_dataset.merge(existent_dataset)
                else:
                    if args.online_order_generation:
                        keyword_arguments = {"combos": self.kl_dict}
                    else:
                        keyword_arguments = {"orders": self.kl_dict}
                    one_piece_of_data, _ = generate_multiple_problems(k, l, num_probs=args.num_probs,
                                                                      train_test="test", backwards=True,
                                                                      transform_gt=args.transform_gt,
                                                                      degree=args.degree,
                                                                      # num_order_or_combo=args.num_order_or_combo,
                                                                      num_order_or_combo=None,
                                                                      **keyword_arguments)

                    self.eval_dataset.merge(one_piece_of_data["all"])
                    self.eval_first_dataset.merge(one_piece_of_data["all_first"])

            self.eval_objectives = set([problem[0]["objectives"][0].name for problem in self.eval_first_dataset])

            print("Eval dataset length ", len(self.eval_dataset))
            print("Eval first step dataset length ", len(self.eval_first_dataset))
            self.reset()

        # Every epoch this is checked, simplified with DataModule by reloading

    def reset(self):
        self.train_dataset = Dataset([])
        self.train_first_dataset = Dataset([])
        for kl in args.train_sets:
            k = kl.split("_")[0][-1]
            l = int(kl[-1])

            if args.online_order_generation:
                keyword_arguments = {"combos": self.kl_dict}
            else:
                keyword_arguments = {"orders": self.kl_dict}

            one_piece_of_data, _ = generate_multiple_problems(k, l, num_probs=args.num_probs,
                                                              train_test="train", backwards=True,
                                                              transform_gt=args.transform_gt,
                                                              degree=args.degree,
                                                              num_order_or_combo=args.num_order_or_combo,
                                                              avoid_objective_names=self.eval_objectives,
                                                              **keyword_arguments)

            self.train_dataset.merge(one_piece_of_data["all"])
            self.train_first_dataset.merge(one_piece_of_data["all_first"])

    def train_dataloader(self):
        # sampler = data_handler.RandomSampler(self.train_dataset)
        # batcher = data_handler.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
        # batch = self.train_dataset.get_multiple(indices=indices)
        # batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=args.obs_mode)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=32, collate_fn=batch_process)

    def val_dataloader(self):
        # sampler = data_handler.RandomSampler(self.train_dataset)
        # batcher = data_handler.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
        # batch = self.train_dataset.get_multiple(indices=indices)
        # batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=args.obs_mode)
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=32, collate_fn=batch_process)

    def test_dataloader(self):
        # sampler = data_handler.RandomSampler(self.train_dataset)
        # batcher = data_handler.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
        # batch = self.train_dataset.get_multiple(indices=indices)
        # batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=args.obs_mode)
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=32, collate_fn=batch_process)


class INTLoop(pl.LightningModule):
    def __init__(self,
                 thm_net,
                 data_module,
                 batch_size=32,
                 lr=1e-4):
        super().__init__()

        self.thm_net = thm_net
        self.lr = lr
        self.batch_size = batch_size
        self.data_module = data_module

    def on_train_start(self):
        train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
            self.test_rollout(self.data_module.train_first_dataset)

        self.log_dict({"train_first_success_rates": train_first_success_rate,
                       "train_first_avg_proof_length": train_first_avg_proof_length})

    def forward(self, batch_states, batch_actions, sl_train=True):
        return self.thm_net(batch_states, batch_actions, sl_train)

    def training_step(self, batch, batch_idx):
        batch_states, batch_actions, batch_name_actions = batch

        log_probs, _, _, (
            lemma_acc, ent_acc, name_acc, diff_lemma_indices, diff_ent_lemma_indices) = self.forward(
            batch_states, batch_actions)

        loss = -log_probs.mean()

        self.log_dict({'loss': loss.detach(),
                       'lemma_acc': lemma_acc.detach(),
                       'ent_acc': ent_acc.detach(),
                       'name_acc': name_acc.detach()})

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_states, batch_actions, batch_name_actions = batch

        log_probs, _, _, \
            (lemma_acc, ent_acc, name_acc, different_lemma_indices, different_ent_lemma_indices) = \
            self.forward(
                batch_states, batch_actions, sl_train=True
            )

        loss = -log_probs.mean()

        self.log_dict({'loss': loss.detach(),
                       'lemma_acc': lemma_acc.detach(),
                       'ent_acc': ent_acc.detach(),
                       'name_acc': name_acc.detach()})

        return

    def test_rollout(self, dataset):

        env_config = {
            "mode": "eval",
            "eval_dataset": dataset.trajectories,
            "online": False,
            "batch_eval": False,
            "verbo": True,
            "obs_mode": args.obs_mode,
            "bag_of_words": args.bag_of_words,
            "time_limit": args.time_limit,
            "degree": args.degree
        }

        success_rate, wrong_cases, success_cases, avg_num_steps = \
            eval_agent(self.thm_net, env_config=env_config)

        return success_rate, wrong_cases, success_cases, avg_num_steps

    def on_train_epoch_end(self):
        if self.current_epoch % args.epoch_per_case_record == 0:
            logging.info("Running rollout..")

            # First-step rollouts
            train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
                self.test_rollout(self.data_module.train_first_dataset)

            self.log_dict({"train_first_success_rates": train_first_success_rate,
                           "train_first_avg_proof_length": train_first_avg_proof_length})

            # val_first_success_rate, val_first_wrong_case, val_first_right_case, val_first_avg_proof_length = \
            #     self.test_rollout(self.data_module.val_first_dataset)

            # self.log_dict({"val_first_success_rates": val_first_success_rate,
            #                "val_first_avg_proof_length": val_first_avg_proof_length})

            test_first_success_rate, test_first_wrong_case, test_first_right_case, test_first_avg_proof_length = \
                self.test_rollout(self.data_module.eval_first_dataset)

            self.log_dict({"test_first_success_rates": test_first_success_rate,
                           "test_first_avg_proof_length": test_first_avg_proof_length})

            cases_record = {
                "train_first_wrong_case": train_first_wrong_case,
                "train_first_right_case": train_first_right_case,
                # "val_first_wrong_case": val_first_wrong_case,
                # "val_first_right_case": val_first_right_case,
                "test_first_wrong_case": test_first_wrong_case,
                "test_first_right_case": test_first_right_case
            }

            json.dump(cases_record,
                      open(
                          os.path.join(
                              args.dump,
                              str(timestamp),
                              "cases_record{0:.0%}.json".format(int(self.global_step / args.updates))),
                          "w")
                      )

            self.data_module.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.1)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            print(f"Error in backward: {e}")


class INTExperiment:
    def __init__(self, config):
        self.config = config
        print(self.config)
        os.makedirs(self.config.exp_config.directory)
        os.makedirs(self.config.checkpoint_dir)

    def run(self):
        torch.set_float32_matmul_precision('high')

        options = dict(
            num_nodes=len(nodename2index),
            num_lemmas=len(thm2index),
            hidden_dim=args.hidden_dim,
            gnn_type=args.gnn_type,
            combined_gt_obj=args.combined_gt_obj,
            attention_type=args.atten_type,
            hidden_layers=args.hidden,
            norm=args.norm,
            entity_cost=args.entity_cost,
            lemma_cost=args.lemma_cost,
            cuda=args.use_gpu,
            attention_heads=args.attention_heads,
            gat_dropout_rate=args.gat_dropout_rate,
            dropout_rate=args.dropout_rate,
        )

        # data_module = INTDataModule(self.config.data_config)
        data_module = INTDataModule()
        # print (next(iter(data_module.train_dataloader())))

        experiment = INTLoop(ThmNet(**options), data_module=data_module)

        logger = WandbLogger(project=self.config.exp_config.logging_config.project,
                             name=self.config.exp_config.name,
                             # config=config_to_dict(self.config),
                             notes=self.config.exp_config.logging_config.notes,

                             # offline=self.config.exp_config.logging_config.offline,
                             offline=True,
                             save_dir=self.config.exp_config.directory,
                             # log_model='all'
                             )

        callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="test_first_success_rate", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{acc}",
                                              # save_on_train_epoch_end=True,
                                              save_last=True,
                                              # save_weights_only=True,
                                              dirpath=self.config.checkpoint_dir)

        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=500,
            callbacks=callbacks,
            accelerator=self.config.exp_config.accelerator,
            devices=[0]
            # devices = self.config.exp_config.device
        )

        # trainer.val_check_interval = self.config.val_frequency
        # if self.config.limit_val_batches:
        #     trainer.limit_val_batches = self.config.val_size // self.config.batch_size

        # experiment.test_rollout(data_module.train_first_dataset)
        # exit()
        trainer.fit(model=experiment, datamodule=data_module)
        logger.experiment.finish()


def main():
    logging_config = LoggingConfig
    logging_config.project = 'none'
    logging_config.offline = True

    cfg = ExperimentConfig
    cfg.logging_config = logging_config
    cfg.experiment = 'test'
    cfg.name = 'int_test'
    cfg.directory = 'experiments/int_test'
    cfg.__post_init__(cfg)

    int_cfg = ThmProvingConfig
    int_cfg.exp_config = cfg
    int_cfg.optimiser_config = OptimiserConfig
    int_cfg.__post_init__(int_cfg)

    #
    # cfg = pyrallis.parse(config_class=ExperimentConfig)
    experiment = INTExperiment(int_cfg)
    experiment.run()


if __name__ == '__main__':
    main()
