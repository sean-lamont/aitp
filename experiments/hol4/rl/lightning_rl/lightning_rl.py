import traceback
from tqdm import tqdm
from experiments.hol4.rl.lightning_rl.rl_data_module import *
from utils.viz_net_torch import make_dot
from models.relation_transformer.relation_transformer_new import AttentionRelations
import einops
from torch_geometric.data import Data
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.gnn.formula_net.formula_net import FormulaNetEdges, message_passing_gnn_induct
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

# import batch_gnn

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
                 config={'max_steps': 50, 'gamma': 0.99, 'lr': 5e-5, 'arg_len': 5},
                 replays="/home/sean/Documents/phd/repo/aitp/data/hol4/replay_test_relation.pk",
                 ):

        super().__init__()
        self.context_net = context_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal

        # todo: more scalable
        if replays is None:
            self.replays = {}
        else:
            try:
                self.replays = torch.load(replays)
            except:
                self.replays = {}

        self.config = config

    def select_goal_fringe(self, history, replay_fringe=None):
        representations, context_set, fringe_sizes = gather_encoded_content_gnn(history, self.encoder_goal,
                                                                                self.device, graph_db=graph_db,
                                                                                token_enc=token_enc)
        context_scores = self.context_net(representations)
        contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
        fringe_scores = []

        for s in scores_by_fringe:
            fringe_score = torch.sum(s)
            fringe_scores.append(fringe_score)

        fringe_scores = torch.stack(fringe_scores)

        fringe_probs = F.softmax(fringe_scores, dim=0)
        fringe_m = Categorical(fringe_probs)

        if replay_fringe is not None:
            fringe = replay_fringe
        else:
            fringe = fringe_m.sample()

        fringe_prob = fringe_m.log_prob(fringe)

        # take the first context in the chosen fringe
        target_context = contexts_by_fringe[fringe][0]

        target_goal = target_context["polished"]["goal"]
        target_representation = representations[context_set.index(target_context)]

        return target_representation, target_goal, fringe, fringe_prob

    def get_tac(self, tac_input):
        tac_probs = self.tac_net(tac_input)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        tac_prob = tac_m.log_prob(tac)
        tac_tensor = tac.to(self.device)
        return tac_tensor, tac_prob

    def get_term_tac(self, target_goal, target_representation, tac, replay_term=None):
        target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)

        arg_probs = []
        tokens = [[t] for t in target_graph.labels if t[0] == "V"]
        token_inds = [i for i, t in enumerate(target_graph.labels) if t[0] == "V"]


        if tokens:
            # pass whole graph through Induct GNN
            induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc).to(
                self.device)
            induct_graph.edge_attr = induct_graph.edge_attr.long()
            induct_nodes = self.induct_net(induct_graph)

            # select representations of Variable nodes with ('V' label only)
            token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(self.device))

            # pass through term_net as before
            target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))
            candidates = torch.cat([token_representations, target_representations], dim=1)
            scores = self.term_net(candidates, tac)
            term_probs = F.softmax(scores, dim=0)
            term_m = Categorical(term_probs.squeeze(1))

            if replay_term is None:
                term = term_m.sample()
            else:
                term = torch.tensor([tokens.index(["V" + replay_term])]).to(self.device)

            arg_probs.append(term_m.log_prob(term))
            tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
            tactic = "Induct_on `{}`".format(tm)

        else:
            arg_probs.append(torch.tensor(0))
            tactic = "Induct_on"

        return tactic, arg_probs

    def get_arg_tac(self, target_representation, num_args, encoded_fact_pool, tac, candidate_args, env, replay_arg=None):
        hidden0 = hidden1 = target_representation
        hidden0 = hidden0.to(self.device)
        hidden1 = hidden1.to(self.device)

        hidden = (hidden0, hidden1)
        # concatenate the candidates with hidden states.

        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
        hiddenl = [hc.unsqueeze(0) for _ in range(num_args)]
        hiddenl = torch.cat(hiddenl)

        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
        candidates = candidates.to(self.device)
        input = tac

        # run it once before predicting the first argument
        hidden, _ = self.arg_net(input, candidates, hidden)

        # the indices of chosen args
        arg_step = []
        arg_step_probs = []

        if tactic_pool[tac] in thm_tactic:
            arg_len = 1
        else:
            arg_len = self.config['arg_len']  # ARG_LEN

        for i in range(arg_len):
            hidden, scores = self.arg_net(input, candidates, hidden)
            arg_probs = F.softmax(scores, dim=0)
            arg_m = Categorical(arg_probs.squeeze(1))

            if replay_arg is None:
                arg = arg_m.sample()
            else:
                if isinstance(replay_arg, list):
                    try:
                        name_parser = replay_arg[i].split(".")
                    except:
                        print(i)
                        print(replay_arg)
                        exit()
                    theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                    theorem_name = name_parser[1]
                    # todo not sure if reverse_database will work...
                    true_arg_exp = reverse_database[(theory_name, theorem_name)]
                else:
                    name_parser = replay_arg.split(".")
                    theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                    theorem_name = name_parser[1]
                    true_arg_exp = reverse_database[(theory_name, theorem_name)]

                arg = torch.tensor(candidate_args.index(true_arg_exp)).to(self.device)

            arg_step.append(arg)
            arg_step_probs.append(arg_m.log_prob(arg))

            hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
            hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

            # encoded chosen argument
            input = encoded_fact_pool[arg].unsqueeze(0)

            # renew candidates
            hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
            hiddenl = [hc.unsqueeze(0) for _ in range(num_args)]
            hiddenl = torch.cat(hiddenl)

            # appends both hidden and cell states (when paper only does hidden?)
            candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
            candidates = candidates.to(self.device)

        tac = tactic_pool[tac]
        arg = [candidate_args[j] for j in arg_step]

        tactic = env.assemble_tactic(tac, arg)
        return tactic, arg_step_probs

    def forward(self, batch):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        encoded_fact_pool = self.encoder_premise(allowed_fact_batch)
        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0
        start_t = time.time()

        for t in range(self.config['max_steps']):
            target_representation, target_goal, fringe, fringe_prob = self.select_goal_fringe(env.history)
            fringe_pool.append(fringe_prob)

            tac, tac_prob = self.get_tac(target_representation)
            tac_pool.append(tac_prob)

            if tactic_pool[tac] in no_arg_tactic:
                tactic = tactic_pool[tac]
                arg_probs = [torch.tensor(0)]

            elif tactic_pool[tac] == "Induct_on":
                tactic, arg_probs = self.get_term_tac(target_goal, target_representation, tac)

            else:
                tactic, arg_probs = self.get_arg_tac(target_representation, len(allowed_arguments_ids),
                                                     encoded_fact_pool, tac, candidate_args, env)

            arg_pool.append(arg_probs)
            action = (fringe.item(), 0, tactic)

            try:
                reward, done = env.step(action)
            except Exception as e:
                print(f"Step exception raised")
                env = HolEnv("T")
                return ("Step error", action)



            if done == True:
                print("Goal Proved in {} steps".format(t + 1))
                # if proved, add to successful replays for this goal
                if env.goal in self.replays.keys():
                    if steps < self.replays[env.goal][0]:
                        print("adding to replay")
                        self.replays[env.goal] = (steps, env.history)
                else:
                    print("Initial add to db...")
                    # print(env.history)
                    if env.history is not None:
                        self.replays[env.goal] = (steps, env.history)
                    else:
                        print("History is none.")
                        print(env.history)
                        print(env)
                break

            if t == self.config['max_steps'] - 1:
                reward = -5
                print("Failed")
                if env.goal in self.replays:
                    return self.run_replay(allowed_fact_batch, allowed_arguments_ids, candidate_args, env, encoded_fact_pool)

            reward_pool.append(reward)
            steps += 1

        return reward_pool, fringe_pool, arg_pool, tac_pool, steps

    def run_replay(self, allowed_fact_batch, allowed_arguments_ids, candidate_args, env, encoded_fact_pool):
        # todo graph replay:
        # reps = self.replays[env.goal]
        # rep_lens = [len(rep[0]) for rep in reps]
        # min_rep = reps[rep_lens.index(min(rep_lens))]
        # known_history, known_action_history, reward_history, _ = min_rep

        print (f"Running replay..")
        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0

        known_history = self.replays[env.goal][1]  # [0]

        for t in range(len(known_history) - 1):
            true_resulting_fringe = known_history[t + 1]
            true_fringe = torch.tensor([true_resulting_fringe["parent"]]).to(self.device)

            target_representation, target_goal, fringe, fringe_prob = self.select_goal_fringe(known_history[:t+1],
                                                                                              replay_fringe=true_fringe)
            fringe_pool.append(fringe_prob)

            tac_input = target_representation.to(self.device)

            tac_probs = self.tac_net(tac_input)
            tac_m = Categorical(tac_probs)
            true_tactic_text = true_resulting_fringe["by_tactic"]

            if true_tactic_text in no_arg_tactic:
                true_tac_text = true_tactic_text
            else:
                tac_args = re.findall(r'(.*?)\[(.*?)\]', true_tactic_text)
                tac_term = re.findall(r'(.*?) `(.*?)`', true_tactic_text)
                tac_arg = re.findall(r'(.*?) (.*)', true_tactic_text)
                if tac_args:
                    true_tac_text = tac_args[0][0]
                    true_args_text = tac_args[0][1].split(", ")
                elif tac_term:  # order matters # TODO: make it irrelavant
                    true_tac_text = tac_term[0][0]
                    true_args_text = tac_term[0][1]
                elif tac_arg:  # order matters because tac_arg could match () ``
                    true_tac_text = tac_arg[0][0]
                    true_args_text = tac_arg[0][1]
                else:
                    true_tac_text = true_tactic_text

            true_tac = torch.tensor([tactic_pool.index(true_tac_text)]).to(self.device)
            tac_pool.append(tac_m.log_prob(true_tac))
            tac = true_tac.to(self.device)
            assert tactic_pool[true_tac.item()] == true_tac_text

            if tactic_pool[true_tac] in no_arg_tactic:
                tactic = tactic_pool[true_tac]
                arg_probs = [torch.tensor(0)]
                arg_pool.append(arg_probs)

            elif tactic_pool[true_tac] == "Induct_on":
                tactic, arg_probs = self.get_term_tac(target_goal, target_representation, tac,
                                                      replay_term=true_args_text)

            else:
                tactic, arg_probs = self.get_arg_tac(target_representation, len(allowed_arguments_ids),
                                         encoded_fact_pool, tac, candidate_args, env, true_args_text)

            arg_pool.append(arg_probs)
            reward = true_resulting_fringe["reward"]
            reward_pool.append(reward)
            steps += 1

            return reward_pool, fringe_pool, arg_pool, tac_pool, steps


    def save_replays(self):
        torch.save(self.replays, "/home/sean/Documents/phd/repo/aitp/data/hol4/replay_test_relation.pk")

    def training_step(self, batch, batch_idx):
        if self.global_step + 1 % 100 == 0:
            print ("Saving...")
            self.save_replays()

        if batch is None:
            print("Error in batch")
            return
        try:
            out = self(batch)

            if len(out) == 2:
                print(f"Error: {out}")
                return

            reward_pool, fringe_pool, arg_pool, tac_pool, steps = out
            loss = self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)
            # g = make_dot(loss)
            # g.view()
            # exit()
            # print (loss)
            return loss

        except Exception as e:
            print("ERROR:")
            print(traceback.print_exc())
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


module = RLData(train_goals=train_goals, test_goals=test_goals, database=compat_db, graph_db=graph_db)
# module.setup("fit")
# batch = next(iter(module.train_dataloader()))
# print (batch)

context_net = ContextPolicy()
tac_net = TacPolicy(len(tactic_pool))
arg_net = ArgPolicy(len(tactic_pool), 256)
term_net = TermPolicy(len(tactic_pool), 256)
induct_net = FormulaNetEdges(1004, 256, 4, global_pool=False, batch_norm=False)

# encoder_premise = FormulaNetEdges(1004, 256, 4, batch_norm=False)
# encoder_goal = FormulaNetEdges(1004, 256, 4, batch_norm=False)

encoder_premise = AttentionRelations(1004, 256)
encoder_goal = AttentionRelations(1004, 256)

experiment = TacticZeroLoop(context_net=context_net, tac_net=tac_net, arg_net=arg_net, term_net=term_net,
                            induct_net=induct_net,
                            encoder_premise=encoder_premise, encoder_goal=encoder_goal)

torch.set_float32_matmul_precision('high')
trainer = pl.Trainer(devices=[1])

trainer.fit(experiment, module)
