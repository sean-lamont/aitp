import traceback
import einops
from torch_geometric.data import Data
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.gnn.formula_net.formula_net import FormulaNetEdges, message_passing_gnn_induct
from torch.utils.data import DataLoader as loader
import lightning.pytorch as pl
import torch.optim

from data.hol4.ast_def import graph_to_torch_labelled
from torch.distributions import Categorical
import torch.nn.functional as F
from datetime import datetime
import pickle
from data.hol4 import ast_def
from torch_geometric.loader import DataLoader
from models.tactic_zero import policy_models
from models.gnn.formula_net import formula_net
import time
from environments.hol4.new_env import *
import numpy as np
#import batch_gnn

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

reverse_database = {(value[0], value[1]) : key for key, value in compat_db.items()}

graph_db = {}

print ("Generating premise graph db...")
for i,t in enumerate(compat_db):
    graph_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc)

with open("data/hol4/data/paper_goals.pk", "rb") as f:
    paper_goals = pickle.load(f)

with open("data/hol4/data/valid_goals_shuffled.pk", "rb") as f:
    valid_goals = pickle.load(f)

train_goals = valid_goals[:int(0.8 * len(valid_goals))]
test_goals = valid_goals[int(0.8 * len(valid_goals)):]

def data_to_relation(batch):
    xis = []
    xjs = []
    edge_attrs = []
    for graph in batch:
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        xi = torch.index_select(x, 0, edge_index[0])
        xj = torch.index_select(x, 0, edge_index[1])
        xis.append(xi)
        xjs.append(xj)
        edge_attrs.append(edge_attr.long())

    xi = torch.nn.utils.rnn.pad_sequence(xis)
    xj = torch.nn.utils.rnn.pad_sequence(xjs)
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attrs)

    mask= (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    return Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)



def gather_encoded_content_gnn(history, encoder, device):
    fringe_sizes = []
    contexts = []
    reverted = []
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        g = revert_with_polish(e)
        reverted.append(g)

    graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc) for t in reverted]

    # if GNN/SAT
    loader = DataLoader(graphs, batch_size = len(reverted))
    batch = next(iter(loader))
    batch.to(device)
    batch.edge_attr = batch.edge_attr.long()#torch.LongTensor(batch.edge_attr)

    representations = torch.unsqueeze(encoder(batch), 1)

    # representations = torch.unsqueeze(encoder(batch.to(device)), 1)

    # if relation transformer
    # graphs = data_to_relation(graphs)
    # graphs.to(device)
    # representations = torch.unsqueeze(encoder(graphs))


    return representations, contexts, fringe_sizes


# todo env reset and gen_fact_batch should be left to DataModule, returns goal and allowed_fact_batch, allowed_args, candidate_args
class RLData(pl.LightningDataModule):
    def __init__(self, train_goals, test_goals, config={'max_steps':50, 'gamma': 0.99}, database=compat_db):
        super().__init__()
        self.config = config
        self.env = HolEnv("T")
        self.train_goals = train_goals
        self.test_goals = test_goals
        self.database = database

    def gen_fact_pool(self, env, goal):
        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', goal[0])))
        goal_theory = self.database[goal[0]][0]
        polished_goal = env.fringe["content"][0]["polished"]["goal"]
        try:
            allowed_arguments_ids = []
            candidate_args = []
            for i, t in enumerate(self.database):
                theory_allowed = self.database[t][0] in allowed_theories
                diff_theory = self.database[t][0] != goal_theory
                prev_theory = int(self.database[t][3]) < int(self.database[polished_goal][3])
                if theory_allowed and (diff_theory or prev_theory):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)
            env.toggle_simpset("diminish", goal_theory)
            print("Removed simpset of {}".format(goal_theory))

        except Exception as e:
            raise Exception(f"Error generating fact pool: {e}")

        # todo: adapt for different data type
        graphs = [graph_db[t] for t in candidate_args]

        loader = DataLoader(graphs, batch_size=len(candidate_args))
        allowed_fact_batch = next(iter(loader))
        allowed_fact_batch.edge_attr = allowed_fact_batch.edge_attr.long()

        # allowed_fact_batch = data_to_relation(graphs)

        #todo not sure if we need allowed_arguments_ids?

        return allowed_fact_batch, allowed_arguments_ids, candidate_args

    def setup_goal(self, goal):
        goal = goal[0]
        self.env.reset(goal[1])
        allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(self.env, goal)
        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, self.env

    def train_dataloader(self):
        return loader(self.train_goals, batch_size=1, collate_fn=self.setup_goal)

    # todo: val set to terminate with??
    def test_dataloader(self):
        return loader(self.test_goals, collate_fn=self.setup_goal)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        allowed_fact_batch = allowed_fact_batch.to(device)

        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env



#  torch lightning initial experiment:
#  define self.{goal_selector, tactic_selector, term_selector, arg_selector}
#  possibly implement as follows: full list of goals to prove is one epoch.
#  then, DataLoader which gives a single goal as batch.
#  Then run forward loop as normal, return loss as defined in updata_parameters

class TacticZeroLoop(pl.LightningModule):
    def __init__(self,
                 context_net,
                 tac_net,
                 arg_net,
                 term_net,
                 induct_net,
                 encoder_premise,
                 encoder_goal,
                 config={'max_steps':50, 'gamma': 0.99, 'lr': 5e-5, 'arg_len': 5},
                 ):

        super().__init__()
        self.context_net = context_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal

        self.config = config

    def forward(self, batch):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch

        encoded_fact_pool = self.encoder_premise(allowed_fact_batch)

        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []

        steps = 0
        replay_flag = False
        start_t = time.time()

        for t in range(self.config['max_steps']):

            ######################################################################################################
            # todo: make this a "get_goal" method
            ######################################################################################################

            # gather all the goals in the history using goal encoder
            try:
                representations, context_set, fringe_sizes = gather_encoded_content_gnn(env.history, self.encoder_goal, self.device)
            except Exception as e:
                print("Encoder error {}".format(e))
                print(traceback.print_exc())
                return ("Encoder error", str(e))

            context_scores = self.context_net(representations)
            contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
            fringe_scores = []

            for s in scores_by_fringe:
                fringe_score = torch.sum(s)
                fringe_scores.append(fringe_score)
            fringe_scores = torch.stack(fringe_scores)

            fringe_probs = F.softmax(fringe_scores, dim=0)
            fringe_m = Categorical(fringe_probs)
            fringe = fringe_m.sample()
            fringe_pool.append(fringe_m.log_prob(fringe))

            # take the first context in the chosen fringe
            try:
                target_context = contexts_by_fringe[fringe][0]
            except Exception as e:
                print("error {} {}".format(contexts_by_fringe, fringe))
                return (f"fringe error...{e}")

            target_goal = target_context["polished"]["goal"]
            target_representation = representations[context_set.index(target_context)]

            ######################################################################################################
            # todo make this a get_tac method
            ######################################################################################################

            tac_input = target_representation  # .unsqueeze(0)
            tac_input = tac_input.to(self.device)

            tac_probs = self.tac_net(tac_input)
            tac_m = Categorical(tac_probs)
            tac = tac_m.sample()
            tac_pool.append(tac_m.log_prob(tac))

            tac_tensor = tac.to(self.device)

            if tactic_pool[tac] in no_arg_tactic:
                tactic = tactic_pool[tac]
                arg_probs = []
                arg_probs.append(torch.tensor(0))
                arg_pool.append(arg_probs)

            ######################################################################################################
            # todo make this a get_term method
            ######################################################################################################

            elif tactic_pool[tac] == "Induct_on":
                target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)#.to(self.device)
                arg_probs = []
                candidates = []

                tokens = [[t] for t in target_graph.labels if t[0] == "V"]
                token_inds = [i for i, t in enumerate(target_graph.labels) if t[0] == "V"]
                if tokens:
                    # pass whole graph through Induct GNN
                    induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc).to(self.device)

                    # induct_nodes = self.induct_net(induct_graph.x.to(self.device),
                    #                                induct_graph.edge_index.to(self.device))

                    induct_graph.edge_attr = induct_graph.edge_attr.long()
                    induct_nodes = self.induct_net(induct_graph)

                    # select representations of Variable nodes with ('V' label only)
                    token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(self.device))
                    # pass through term_net as before

                    # target_representation_list = [target_representation for _ in tokens]
                    # target_representations = torch.cat(target_representation_list)


                    target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))


                    candidates = torch.cat([token_representations, target_representations], dim=1)
                    # candidates = candidates.to(self.device)
                    scores = self.term_net(candidates, tac_tensor)
                    term_probs = F.softmax(scores, dim=0)
                    try:
                        term_m = Categorical(term_probs.squeeze(1))
                    except:
                        print("probs: {}".format(term_probs))
                        print("candidates: {}".format(candidates.shape))
                        print("scores: {}".format(scores))
                        print("tokens: {}".format(tokens))
                        exit()
                    term = term_m.sample()
                    arg_probs.append(term_m.log_prob(term))
                    # induct_arg.append(tokens[term])
                    tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
                    arg_pool.append(arg_probs)
                    if tm:
                        tactic = "Induct_on `{}`".format(tm)
                        # print (tactic)
                    else:
                        print("tm is empty")
                        print(tokens)
                        # only to raise an error
                        tactic = "Induct_on"
                else:
                    arg_probs.append(torch.tensor(0))
                    # induct_arg.append("No variables")
                    arg_pool.append(arg_probs)
                    tactic = "Induct_on"

            ######################################################################################################
            # todo make this a get_arg method
            ######################################################################################################
            else:
                hidden0 = hidden1 = target_representation
                hidden0 = hidden0.to(self.device)
                hidden1 = hidden1.to(self.device)

                hidden = (hidden0, hidden1)

                # concatenate the candidates with hidden states.
                hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]
                try:
                    hiddenl = torch.cat(hiddenl)
                except Exception as e:
                    print (f"Error in hiddenl: {e}")
                    return ("hiddenl error...{}", str(e))

                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(self.device)
                input = tac_tensor

                # run it once before predicting the first argument
                hidden, _ = self.arg_net(input, candidates, hidden)

                # the indices of chosen args
                arg_step = []
                arg_step_probs = []

                if tactic_pool[tac] in thm_tactic:
                    arg_len = 1
                else:
                    arg_len = self.config['arg_len'] # ARG_LEN

                for _ in range(arg_len):
                    hidden, scores = self.arg_net(input, candidates, hidden)
                    arg_probs = F.softmax(scores, dim=0)
                    arg_m = Categorical(arg_probs.squeeze(1))
                    arg = arg_m.sample()
                    arg_step.append(arg)
                    arg_step_probs.append(arg_m.log_prob(arg))

                    hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
                    hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

                    # encoded chosen argument
                    input = encoded_fact_pool[arg].unsqueeze(0)  # .unsqueeze(0)

                    # renew candidates
                    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    hiddenl = torch.cat(hiddenl)
                    # appends both hidden and cell states (when paper only does hidden?)
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(self.device)

                arg_pool.append(arg_step_probs)

                tac = tactic_pool[tac]
                arg = [candidate_args[j] for j in arg_step]

                ######################################################################################################
                ######################################################################################################

                tactic = env.assemble_tactic(tac, arg)

            action = (fringe.item(), 0, tactic)

            try:
                reward, done = env.step(action)
            except Exception as e:
                print(f"Step exception raised. {e}")
                # exit()
                return ("Step error", action)
                print("Handling: {}".format(env.handling))
                print("Using: {}".format(env.using))
                # try again
                # counter = env.counter
                frequency = env.frequency
                env.close()
                print("Aborting current game ...")
                print("Restarting environment ...")
                print(env.goal)
                env = HolEnv(env.goal)
                flag = False
                break

            if t == self.config['max_steps'] - 1:
                reward = -5

            reward_pool.append(reward)
            steps += 1

            # todo: replays
            if done == True:
                print("Goal Proved in {} steps".format(t + 1))
                # # iteration_rewards.append(total_reward)
                #
                # # if proved, add to successful replays for this goal
                # if env.goal in self.replays.keys():
                #     # if proof done in less steps than before, add to dict
                #     if steps < self.replays[env.goal][0]:
                #         print("adding to replay")
                #         print(env.history)
                #         self.replays[env.goal] = (steps, env.history)
                # else:
                #
                #     print("Initial add to db...")
                #     print(env.history)
                #     if env.history is not None:
                #         self.replays[env.goal] = (steps, env.history)
                #
                #     else:
                #         print("History is none.")
                #         print(env.history)
                #         print(env)
                break

            if t == self.config['max_steps'] - 1:
                print("Failed.")
                # if env.goal in self.replays.keys():
                #     replay_flag = True
                #
                #     return trace, steps, done, 0, 0, replay_flag

        return reward_pool, fringe_pool, arg_pool, tac_pool, steps


        # return trace, steps, done, 0, float(np.sum(reward_print)), replay_flag

    def training_step(self, batch, batch_idx):
        # todo use self forward for training step, if replay, have new method for replay in place of forward
        # todo add probability to outputs
        try:
            reward_pool, fringe_pool, arg_pool, tac_pool, steps = self(batch)
            loss = self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)
            return loss
        except Exception as e:
            print ("error")
            print (traceback.print_exc())
            return



    def update_params(self, reward_pool, fringe_pool, arg_pool, tac_pool, steps):
        print("Updating parameters ... ")
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


module = RLData(train_goals = train_goals, test_goals=test_goals)
# module.setup("fit")

# batch = next(iter(module.train_dataloader()))
# print (batch)

context_net = ContextPolicy()
tac_net = TacPolicy(len(tactic_pool))
arg_net = ArgPolicy(len(tactic_pool), 256)
term_net = TermPolicy(len(tactic_pool), 256)
induct_net = FormulaNetEdges(1000, 256, 4, global_pool=False, batch_norm=False)
encoder_premise = FormulaNetEdges(1000, 256, 4, batch_norm=False)
encoder_goal = FormulaNetEdges(1000, 256, 4, batch_norm=False)

experiment = TacticZeroLoop(context_net=context_net, tac_net=tac_net, arg_net=arg_net, term_net=term_net, induct_net=induct_net,
                            encoder_premise=encoder_premise, encoder_goal=encoder_goal)

torch.set_float32_matmul_precision('high')
trainer = pl.Trainer(devices=1)
trainer.fit(experiment, module)