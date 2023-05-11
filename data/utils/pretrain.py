from data.utils.dataset import H5DataModule, MongoDataModule
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
import itertools
import random
from torch.nn.utils.rnn import pad_sequence
from utils.mongodb_utils import get_batches, get_all_batches
from tqdm import tqdm
import traceback
import wandb
from lightning.pytorch.loggers import WandbLogger
from pymongo import MongoClient
from models import gnn_edge_labels, inner_embedding_network
from models.get_model import get_model
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import DataLoader as StandardLoader
from torch.utils.data import TensorDataset


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


# todo Better way for softmax_idx, combined graph setup, positonal encoding, model saving/versioning, parametrised classifier, DAGLSTM?

class PremiseSelection(pl.LightningModule):
    def __init__(self, embedding_model_goal, embedding_model_premise, classifier, lr=1e-4):
        super().__init__()
        self.embedding_model_goal = embedding_model_goal
        self.embedding_model_premise = embedding_model_premise
        self.classifier = classifier
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = 32

    def forward(self, goal, premise):
        embedding_goal = self.embedding_model_goal(goal)
        embedding_premise = self.embedding_model_premise(premise)
        preds = self.classifier(torch.cat([embedding_goal, embedding_premise], dim = 1))
        preds = torch.clip(preds, self.eps, 1 - self.eps)
        return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        # goal, premise, y = batch[0] #batch[0][0], batch[0][1], batch[0][2]
        goal, premise, y = batch #batch[0][0], batch[0][1], batch[0][2]
        # goal, premise, y = batch
        preds = self(goal, premise)
        # print (preds.shape, y.shape)
        # exit()
        loss = binary_loss(preds, y)
        # loss = torch.nn.functional.cross_entropy(preds, y)
        self.log("loss", loss, batch_size = self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # if self.global_step == 0:
            # self.logger.define_metric('val_accuracy', summary='max')

        # goal, premise, y = batch
        # goal, premise, y = batch[0]#batch[0][0] batch[0][1], batch[0][2]
        goal, premise, y = batch #batch[0][0], batch[0][1], batch[0][2]
        preds =  self(goal, premise)
        preds = (preds > 0.5)
        acc = torch.sum(preds == y) / y.size(0)
        self.log("acc", acc, batch_size = self.batch_size)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer

class PremiseSelectionExperiment:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model_config']
        self.data_config = config['data_config']
        self.exp_config = config['exp_config']

        self.data_options = self.data_config['data_options']
        self.source_config = self.data_config['source_config']
        self.embedding_dim = self.model_config['embedding_dim']
        self.lr = self.exp_config['learning_rate']
        self.weight_decay = self.exp_config['weight_decay']
        self.epochs = self.exp_config['epochs']
        self.batch_size = self.exp_config['batch_size']
        self.save = self.exp_config['model_save']
        self.val_size = self.exp_config['val_size']
        self.logging = self.exp_config['logging']
        self.device = self.exp_config['device']
        self.max_errors = self.exp_config['max_errors']
        self.val_frequency = self.exp_config['val_frequency']

'''
Premise selection experiment with separate encoders for goal and premise
'''
class SeparateEncoderPremiseSelection(PremiseSelectionExperiment):
    def __int__(self, config):
        super(self, config)
        self.save_hyperparameters()

    def run_lightning(self):
        torch.set_float32_matmul_precision('high')
        ps = PremiseSelection(get_model(self.model_config), get_model(self.model_config), gnn_edge_labels.F_c_module_(self.embedding_dim * 2))

        data_module = H5DataModule(config = {'data_dir': '/home/sean/Documents/phd/repo/aitp/data/utils/processed_data'})

        # config = {'buf_size': 128, 'batch_size': 32, 'db_name': 'hol_step',
        #           'collection_name': 'pretrain_graphs', 'options': ['edge_attr', 'edge_index', 'softmax_idx']}
        # data_module = MongoDataModule(config)

        # todo logging/wandb integration, model checkpoints

        logger = WandbLogger(project=self.config['project'],
                             name=self.config['name'],
                             )

        trainer = pl.Trainer(val_check_interval=1000,
                             limit_val_batches=self.val_size // self.batch_size,
                             # max_steps=1000,
                             logger=logger,
                             enable_progress_bar=True,
                             log_every_n_steps=500,
                             accelerator='gpu',
                             devices=2,
                             strategy='ddp',
                             # profiler='pytorch',
                             enable_checkpointing=False)


        trainer.fit(model=ps, datamodule=data_module)



# criterion = torch.nn.CrossEntropyLoss()
# class MaskPretrain(PremiseSelectionExperiment):
#     def __int__(self, config):
#         super(self, config)
#
#     def run_mask_experiment(self):
#         self.graph_net = self.get_model().to(self.device)
#         print("Model details:")
#
#         print(self.graph_net)
#
#         if self.logging:
#             wandb.log({"Num_model_params": sum([p.numel() for p in self.graph_net.parameters() if p.requires_grad])})
#
#         fc = torch.nn.Sequential(torch.nn.Linear(self.model_config['embedding_dim'],self.model_config['embedding_dim']),
#                                             torch.nn.ReLU(),
#                                             torch.nn.LayerNorm(self.model_config['embedding_dim']),
#                                             torch.nn.Linear(self.model_config['embedding_dim'], self.model_config['vocab_size'])).to(self.device)
#
#         op_g = torch.optim.AdamW(self.graph_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         op_fc = torch.optim.AdamW(fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         training_losses = []
#
#         val_losses = []
#         best_acc = 0.
#
#         graph_collection, split_collection = get_data(self.source_config)
#
#         val_cursor = split_collection.find({"split":"valid"}).limit(self.val_size)
#
#         for j in range(self.epochs):
#             print(f"Epoch: {j}")
#             err_count = 0
#
#             # train_cursor = split_collection.aggregate([{"$match": {"split": "train"}}, {"$sample": {"size": 10000000}}])
#             train_cursor = graph_collection.find()
#             batches = get_batches(train_cursor, self.batch_size)
#
#             for i,db_batch in tqdm(enumerate(batches)):
#                 print (len(db_batch))
#                 try:
#                     # batch = to_batch_graph(db_batch, graph_collection, self.data_options)
#                     batch = self.get_batch(db_batch, self.data_options)
#                 except Exception as e:
#                     print(f"Error in batch: {e}")
#                     traceback.print_exc()
#                     continue
#
#                 # print (batch)
#                 op_g.zero_grad()
#                 op_fc.zero_grad()
#
#                 try:
#                     masked_encs = self.graph_net(batch.to(self.device))
#
#                     preds = fc(masked_encs)
#
#                     eps = 1e-6
#
#                     preds = torch.clip(preds, eps, 1 - eps)
#
#                     loss = criterion(torch.flatten(preds), batch.mask_idx.to(self.device))
#
#                     loss.backward()
#
#                     # op_enc.step()
#                     op_g.step()
#                     op_fc.step()
#
#
#                 except Exception as e:
#                     err_count += 1
#                     if err_count > self.max_errors:
#                         return Exception("Too many errors in training")
#                     print(f"Error in training {e}")
#                     traceback.print_exc()
#                     continue
#
#                 training_losses.append(loss.detach() / self.batch_size)
#
#                 if i % self.val_frequency == 0:
#                     print (sum(training_losses[-100:]) / len(training_losses[-100:]))
#



# todo add logic here for loading larger batch into memory, create dataloader from this then yield the next value in loader until done
# todo this should reduce the required number of queries
# def test_iter(batches, batch_fn, graph_collection, options):
#
#     for batch in batches:
#         # print (batch[0])
#
#         # hack it as list for now
#         batch = list(batch)
#         # print (batch[0])
#
#         stmts = list(set([sample['stmt'] for sample in batch]))
#         conjs = list(set([sample['conj'] for sample in batch]))
#
#         stmts.extend(conjs)
#
#         exprs = list(graph_collection.find({"_id": {"$in" : stmts}}))
#
#         expr_dict = {expr["_id"]: expr["graph"] for expr in exprs}
#
#         # print (len(expr_dict))
#         # print (list(expr_dict.keys())[0])
#
#
#         for i in range(0, len(batch), 32):
#             if (i + 1) * 32 >= len(batch):
#                 # print (f"fuck you cunt {i}")
#                 break
#             # print (batch_fn(batch[i * 32: (i + 1) * 32], graph_collection, options, expr_dict))
#             # exit()
#             yield batch_fn(batch[i * 32: (i + 1) * 32], graph_collection, options, expr_dict)



    # def run_dual_encoders(self):
    #     self.graph_net_1 = self.get_model().to(self.device)
    #     self.graph_net_2 = self.get_model().to(self.device)
    #
    #     print("Model details:")
    #
    #     print(self.graph_net_1)
    #
    #     if self.logging:
    #         wandb.log({"Num_model_params": sum([p.numel() for p in self.graph_net_1.parameters() if p.requires_grad])})
    #
    #     fc = gnn_edge_labels.F_c_module_(self.embedding_dim * 2).to(self.device)
    #
    #     op_g1 = torch.optim.AdamW(self.graph_net_1.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #
    #     op_g2 = torch.optim.AdamW(self.graph_net_2.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #
    #     op_fc = torch.optim.AdamW(fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #
    #     training_losses = []
    #
    #     data_module = H5DataModule(config={})
    #
    #     data_module.setup('fit')
    #
    #     loader = data_module.train_dataloader()
    #
    #
    #     for i, batch in tqdm(enumerate(loader)):
    #         data_1, data_2, y = batch[0]
    #
    #         err_count = 0
    #
    #         op_g1.zero_grad()
    #         op_g2.zero_grad()
    #         op_fc.zero_grad()
    #
    #         try:
    #
    #             graph_enc_1 = self.graph_net_1(data_1.to(self.device))
    #             graph_enc_2 = self.graph_net_2(data_2.to(self.device))
    #
    #             preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
    #
    #             eps = 1e-6
    #
    #             preds = torch.clip(preds, eps, 1 - eps)
    #
    #             loss = torch.nn.functional.cross_entropy(torch.flatten(preds), y.to(self.device).float())
    #
    #             loss.backward()
    #
    #             # op_enc.step()
    #             op_g1.step()
    #             op_g2.step()
    #             op_fc.step()
    #
    #
    #         except Exception as e:
    #             err_count += 1
    #             if err_count > self.max_errors:
    #                 return Exception("Too many errors in training")
    #             print(f"Error in training {e}")
    #             traceback.print_exc()
    #             continue
    #
    #         training_losses.append(loss.detach() / self.batch_size)
    #
    #         if i % 1000 == 0:
    #             print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

    #
    #         self.graph_net_1.eval()
    #         self.graph_net_2.eval()
    #
    #         val_count = []
    #
    #         val_cursor.rewind()
    #
    #         get_val_batches = get_batches(val_cursor, self.batch_size)
    #
    #         for db_val in get_val_batches:
    #             val_err_count = 0
    #             try:
    #                 data_1, data_2, y = self.get_batch(db_val, graph_collection, self.data_options)
    #
    #                 validation_loss = self.val_acc_dual_encoder(self.graph_net_1, self.graph_net_2, data_1,
    #                                                             data_2, y, fc, self.device)
    #
    #                 val_count.append(validation_loss.detach())
    #
    #             except Exception as e:
    #                 print(f"Error {e}, batch:")
    #                 val_err_count += 1
    #                 traceback.print_exc()
    #                 continue
    #
    #         validation_loss = (sum(val_count) / len(val_count)).detach()
    #         val_losses.append((validation_loss, j, i))
    #
    #         print(
    #             "Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
    #
    #         print("Val acc: {}".format(validation_loss.detach()))
    #
    #         print(f"Failed batches: {err_count}")
    #
    #         if self.logging:
    #             wandb.log({"acc": validation_loss.detach(),
    #                        "train_loss_avg": sum(training_losses[-100:]) / len(training_losses[-100:]),
    #                        "epoch": j})
    #
    #         if validation_loss > best_acc:
    #             best_acc = validation_loss
    #             print(f"New best validation accuracy: {best_acc}")
    #             # only save encoder if best accuracy so far
    #
    #             if self.save == True:
    #                 torch.save(self.graph_net_1, self.exp_config['model_dir'] + "/gnn_transformer_goal_hol4")
    #                 torch.save(self.graph_net_2, self.exp_config['model_dir'] + "/gnn_transformer_premise_hol4")
    #
    #         self.graph_net_1.train()
    #         self.graph_net_2.train()
    #
    # if self.logging:
    #     wandb.log({"failed_batches": err_count})

#     return
#
#
# def val_acc_dual_encoder(self, model_1, model_2, data_1, data_2, y, fc, device):
#
#     # data_1, data_2,y = self.get_batch(batch, data_options)
#
#     graph_enc_1 = model_1(data_1.to(device))
#
#     graph_enc_2 = model_2(data_2.to(device))
#
#     preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds > 0.5).long()
#
#     return torch.sum(preds == y.to(device)) / y.size(0)
#
