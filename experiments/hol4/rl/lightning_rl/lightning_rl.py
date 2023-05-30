import traceback
import logging
from models.get_model import get_model
from lightning.pytorch.callbacks import ModelCheckpoint
from experiments.hol4.rl.lightning_rl.agent_utils import *
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
from experiments.hol4.rl.lightning_rl.rl_data_module import *
from utils.viz_net_torch import make_dot
from models.relation_transformer.relation_transformer_new import AttentionRelations
import einops
from torch_geometric.data import Data
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.gnn.formula_net.formula_net import FormulaNetEdges, message_passing_gnn_induct
from models.sat.models import GraphTransformer
import lightning.pytorch as pl
import torch.optim
from data.hol4.ast_def import graph_to_torch_labelled
from torch.distributions import Categorical
import torch.nn.functional as F
from datetime import datetime
import pickle
from data.hol4 import ast_def
from torch_geometric.loader import DataLoader
import time
from environments.hol4.new_env import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')



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

tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic


def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions):
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)
    return goal


def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs


# todo DB data format/ HDF5 etc...

with open("data/hol4/data/torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("data/hol4/data/graph_token_encoder.pk", "rb") as f:
    token_enc = pickle.load(f)

encoded_graph_db = []
with open('data/hol4/data/adjusted_db.json') as f:
    compat_db = json.load(f)

reverse_database = {(value[0], value[1]): key for key, value in compat_db.items()}

graph_db = {}

print("Generating premise graph db...")
for i, t in enumerate(compat_db):
    graph_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc)

with open("data/hol4/data/paper_goals.pk", "rb") as f:
    paper_goals = pickle.load(f)

with open("data/hol4/data/valid_goals_shuffled.pk", "rb") as f:
    valid_goals = pickle.load(f)

train_goals = valid_goals[:int(0.8 * len(valid_goals))]
test_goals = valid_goals[int(0.8 * len(valid_goals)):]

"""
 Torch Lightning TacticZero Loop:
"""

class TacticZeroLoop(pl.LightningModule):
    def __init__(self,
                 context_net,
                 tac_net,
                 arg_net,
                 term_net,
                 induct_net,
                 encoder_premise,
                 encoder_goal,
                 config,
                 ):

        super().__init__()
        self.context_net = context_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal
        self.proven = []
        self.cumulative_proven = []
        self.replayed = 0

        self.config = config
        self.dir = self.config['dir_path']
        os.makedirs(self.dir, exist_ok=True)

        logging.basicConfig(filename=self.dir + 'log', level=logging.DEBUG)
        
        # todo replay DB + log probs
        if "replay_dir" in self.config:
            self.replays = torch.load(self.config['replay_dir'])
            self.replay_dir = self.config['replay_dir']
        else:
            print(f"Creating new replay dir {self.dir + 'replays'}")
            self.replays = {}
            self.replay_dir = self.dir + 'replays'

    def forward(self, batch):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        encoded_fact_pool = self.encoder_premise(allowed_fact_batch)
        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0
        start_t = time.time()

        # todo stats per goal? save to experiment directory?

        # todo 1 - eps replay? I.e run replay only with prob 1 - eps
        for t in range(self.config['max_steps']):
            target_representation, target_goal, fringe, fringe_prob = select_goal_fringe(history=env.history,
                                                                                         encoder_goal=self.encoder_goal,
                                                                                         graph_db=graph_db,
                                                                                         token_enc=token_enc,
                                                                                         context_net=self.context_net,
                                                                                         device=self.device,
                                                                                        data_type = self.config['data_type'])
            fringe_pool.append(fringe_prob)
            tac, tac_prob = get_tac(tac_input=target_representation,
                                    tac_net=self.tac_net,
                                    device=self.device)

            tac_pool.append(tac_prob)

            if tactic_pool[tac] in no_arg_tactic:
                tactic = tactic_pool[tac]
                arg_probs = [torch.tensor(0)]

            elif tactic_pool[tac] == "Induct_on":
                tactic, arg_probs = get_term_tac(target_goal=target_goal,
                                                 target_representation=target_representation,
                                                 tac=tac,
                                                 term_net=self.term_net,
                                                 induct_net=self.induct_net,
                                                 device=self.device,
                                                 token_enc=token_enc)

            else:
                tactic, arg_probs = get_arg_tac(target_representation=target_representation,
                                                num_args=len(allowed_arguments_ids),
                                                encoded_fact_pool=encoded_fact_pool,
                                                tac=tac,
                                                candidate_args=candidate_args,
                                                env=env,
                                                device=self.device,
                                                arg_net=self.arg_net,
                                                arg_len=self.config['arg_len'],
                                                reverse_database=reverse_database)

            arg_pool.append(arg_probs)
            action = (fringe.item(), 0, tactic)

            try:
                reward, done = env.step(action)
            except Exception as e:
                # print(f"Step exception raised with: {action}, continuing..")
                # todo negative reward, do we need to reset env at all??
                # env = HolEnv("T")
                # return ("Step error", action)
                reward = -1
                done = False

            reward_pool.append(reward)
            steps += 1

            if done == True:
                # print("Goal Proved in {} steps".format(t + 1))
                self.proven.append([env.polished_goal[0], t + 1])
                self.log("cur_proved", len(self.proven), prog_bar=True)

                if env.goal in self.replays.keys():
                    if steps < self.replays[env.goal][0]:
                        # print("Adding to replay")
                        self.replays[env.goal] = (steps, env.history)
                else:
                    self.cumulative_proven.append([env.polished_goal[0]])
                    # print("Initial add to db...")
                    if env.history is not None:
                        self.replays[env.goal] = (steps, env.history)
                    else:
                        print("History is none.")
                        print(env.history)
                        print(env)

                # reward_pool.append(reward)
                # steps += 1
                break

            if t == self.config['max_steps'] - 1:
                reward = -5
                # print("Failed")
                if env.goal in self.replays:
                    self.replayed += 1
                    self.log("cur_replayed", self.replayed, prog_bar=True)
                    return self.run_replay(allowed_arguments_ids, candidate_args, env, encoded_fact_pool)

            # reward_pool.append(reward)
            # steps += 1

        return reward_pool, fringe_pool, arg_pool, tac_pool, steps

    def run_replay(self, allowed_arguments_ids, candidate_args, env, encoded_fact_pool):
        # todo graph replay:
        # reps = self.replays[env.goal]
        # rep_lens = [len(rep[0]) for rep in reps]
        # min_rep = reps[rep_lens.index(min(rep_lens))]
        # known_history, known_action_history, reward_history, _ = min_rep

        # print (f"Running replay..")
        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0

        known_history = self.replays[env.goal][1]

        for t in range(len(known_history) - 1):
            true_resulting_fringe = known_history[t + 1]
            true_fringe = torch.tensor([true_resulting_fringe["parent"]]).to(self.device)

            target_representation, target_goal, fringe, fringe_prob = select_goal_fringe(history=known_history[:t+1],
                                                                                              encoder_goal=self.encoder_goal,
                                                                                              graph_db=graph_db,
                                                                                              token_enc=token_enc,
                                                                                              context_net=self.context_net,
                                                                                              device=self.device,
                                                                                              replay_fringe=true_fringe,
                                                                                              data_type=self.config['data_type'])
            fringe_pool.append(fringe_prob)
            tac_probs = self.tac_net(target_representation)
            tac_m = Categorical(tac_probs)

            true_tactic_text = true_resulting_fringe["by_tactic"]
            true_tac_text, true_args_text = get_replay_tac(true_tactic_text)

            true_tac = torch.tensor([tactic_pool.index(true_tac_text)]).to(self.device)
            tac_pool.append(tac_m.log_prob(true_tac))

            assert tactic_pool[true_tac.item()] == true_tac_text

            if tactic_pool[true_tac] in no_arg_tactic:
                arg_probs = [torch.tensor(0)]
                arg_pool.append(arg_probs)

            elif tactic_pool[true_tac] == "Induct_on":
                _, arg_probs = get_term_tac(target_goal=target_goal,
                                                 target_representation=target_representation,
                                                 tac=true_tac,
                                                 term_net=self.term_net,
                                                 induct_net=self.induct_net,
                                                 device=self.device,
                                                 token_enc=token_enc,
                                                 replay_term=true_args_text)
            else:
                _, arg_probs = get_arg_tac(target_representation=target_representation,
                                                num_args=len(allowed_arguments_ids),
                                                encoded_fact_pool=encoded_fact_pool,
                                                tac=true_tac,
                                                candidate_args=candidate_args,
                                                env=env,
                                                device=self.device,
                                                arg_net=self.arg_net,
                                                arg_len=self.config['arg_len'],
                                                reverse_database=reverse_database,
                                                replay_arg=true_args_text)

            arg_pool.append(arg_probs)
            reward = true_resulting_fringe["reward"]
            reward_pool.append(reward)
            steps += 1

        # print (f"Replay took {steps}")
        return reward_pool, fringe_pool, arg_pool, tac_pool, steps


    def save_replays(self):
        torch.save(self.replays, self.replay_dir)

    def training_step(self, batch, batch_idx):
        if batch is None:
            logging.debug("Error in batch")
            return
        try:
            out = self(batch)
            if len(out) == 2:
                # print (f"error in run {out}")
                logging.debug(f"Error in run: {out}")
                return

            reward_pool, fringe_pool, arg_pool, tac_pool, steps = out
            loss = self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)

            if type(loss) != torch.Tensor:
                # print (f"error in loss {loss}")
                logging.debug(f"Error in loss: {loss}")
                return

            # g = make_dot(loss)
            # g.view()
            # loss.backward()
            # print ([(name, param.grad, torch.nonzero(param.grad).shape[0]) for name, param in self.named_parameters() if "initial_encoder" in name ])
            # exit()

            return loss

        except:
            logging.debug(f"Error in training: {traceback.print_exc()}")
            return

    def on_train_epoch_end(self):
        self.log_dict({"epoch_proven": len(self.proven),
                        "cumulative_proven": len(self.cumulative_proven)},
                    prog_bar=True)

        # todo logging goals, steps etc. proven...
        # self.logger.log_text(key="Total Proved", columns = ["Goals"], data=self.cumulative_proven)
        # self.logger.log_text(key="Epoch Proved", columns = ["Goals", "Steps"], data=self.proven)

        self.proven = []
        self.replayed = 0
        self.save_replays()

    def update_params(self, reward_pool, fringe_pool, arg_pool, tac_pool, steps):
        # print("Updating parameters ... ")
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.config['gamma'] + reward_pool[i]
                reward_pool[i] = running_add

        total_loss = 0
        for i in range(steps):
            reward = reward_pool[i]
            fringe_loss = -fringe_pool[i] * (reward)
            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            tac_loss = -tac_pool[i] * (reward)
            loss = fringe_loss + tac_loss + arg_loss
            total_loss += loss

        return total_loss

    # todo?
    # def validation_step(self):

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config['lr'])
        return optimizer

def get_model_dict(prefix, state_dict):
    return {k[len(prefix)+1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}

# hack for now to deal with BatchNorm Loading
def get_model_dict_fn(model, prefix, state_dict):
    ret_dict = {}
    own_state = model.state_dict()
    for k,v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix)+1:]
            if "mlp.3" in k:
                k = k.replace('3', '2')
            if k not in own_state:
                continue
            ret_dict[k] = v

    return ret_dict

torch.set_float32_matmul_precision('high')


VOCAB_SIZE = 1004
EMBEDDING_DIM = 256

context_net = ContextPolicy()
tac_net = TacPolicy(len(tactic_pool))
arg_net = ArgPolicy(len(tactic_pool), EMBEDDING_DIM)
term_net = TermPolicy(len(tactic_pool), EMBEDDING_DIM)
induct_net = FormulaNetEdges(VOCAB_SIZE, EMBEDDING_DIM, 3, global_pool=False, batch_norm=False)


# SAT
sat_config = {
    "model_type": "sat",
    "num_edge_features":  200,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "dim_feedforward": 256,
    "num_heads": 2,
    "num_layers": 2,
    "in_embed": True,
    "se": "formula-net",
    "abs_pe": False,
    "abs_pe_dim": 256,
    "use_edge_attr": True,
    "dropout": 0.,
    "gnn_layers": 3,
    "directed_attention": False,
}
encoder_premise = get_model(sat_config)
encoder_goal = get_model(sat_config)
ckpt_dir = "/home/sean/Documents/phd/repo/aitp/experiments/hol4/supervised/model_checkpoints/sat_91_24.ckpt"
ckpt = torch.load(ckpt_dir)['state_dict']
encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))

# relation
# encoder_premise = AttentionRelations(VOCAB_SIZE, EMBEDDING_DIM)
# encoder_goal = AttentionRelations(VOCAB_SIZE, EMBEDDING_DIM)
# ckpt_dir = "/home/sean/Documents/phd/repo/aitp/sat/hol4/supervised/model_checkpoints/epoch=5-step=41059.ckpt"
# ckpt = torch.load(ckpt_dir)['state_dict']
# encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
# encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))
#

# GNN
# encoder_premise = FormulaNetEdges(VOCAB_SIZE, EMBEDDING_DIM, 3, batch_norm=False)
# encoder_goal = FormulaNetEdges(VOCAB_SIZE, EMBEDDING_DIM, 3, batch_norm=False)
# ckpt_dir = "/home/sean/Documents/phd/repo/aitp/experiments/hol4/supervised/model_checkpoints/formula_net_best_91.ckpt"
# ckpt = torch.load(ckpt_dir)['state_dict']
# encoder_premise.load_state_dict(get_model_dict_fn(encoder_premise, 'embedding_model_premise', ckpt))
# encoder_goal.load_state_dict(get_model_dict_fn(encoder_goal, 'embedding_model_goal', ckpt))
# # encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
# # encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))
#

# notes = "gnn_5_step/"
# notes = "relation_5_step/"
notes = "sat_5_step/"

save_dir = '/home/sean/Documents/phd/repo/aitp/experiments/hol4/rl/lightning_rl/experiments/' + notes

config = {'max_steps': 5,
          'gamma': 0.99,
          'lr': 5e-5,
          'arg_len': 5,
          'data_type': 'graph',
          # 'replay_dir': '/home/sean/Documents/phd/repo/aitp/data/hol4/replay_test_gnn.pk',
          'dir_path': save_dir}


module = RLData(train_goals=train_goals, test_goals=test_goals, database=compat_db, graph_db=graph_db,
                config=config)


experiment = TacticZeroLoop(context_net=context_net, tac_net=tac_net, arg_net=arg_net, term_net=term_net,
                            induct_net=induct_net,
                            encoder_premise=encoder_premise, encoder_goal=encoder_goal, config=config)

logger = WandbLogger(project="RL Test",
                     name="TacticZero SAT 5 step",
                     config=config,
                     notes=notes,
                     offline=False)

callbacks = []
checkpoint_callback = ModelCheckpoint(monitor="epoch_proven", mode="max", auto_insert_metric_name=True,
                                      save_weights_only=True, dirpath=save_dir)

callbacks.append(checkpoint_callback)

trainer = pl.Trainer(devices=[1],
                     logger=logger,
                     callbacks=callbacks)

trainer.fit(experiment, module)