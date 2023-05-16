import json
import traceback
import torch
from data.hol4.ast_def import graph_to_torch_labelled
from torch.distributions import Categorical
import torch.nn.functional as F
from datetime import datetime
import pickle
from data.hol4 import ast_def
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import utp_model, inner_embedding_network
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


'''

High level agent class 

'''
class Agent:
    def __init__(self, tactic_pool):
        self.tactic_pool = tactic_pool    
        self.load_encoder()
    
    def load_agent(self):
        pass
    
    def load_encoder(self):
        pass
        
    def run(self, env, max_steps):
        pass
    
    def update_params(self):
        pass

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

def gather_encoded_content_gnn(history, encoder):
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

    loader = DataLoader(graphs, batch_size = len(reverted))

    batch = next(iter(loader))

    batch.edge_attr = batch.edge_attr.long()#torch.LongTensor(batch.edge_attr)

    representations = torch.unsqueeze(encoder(batch.to(device)), 1)

    return representations, contexts, fringe_sizes


'''

Torch implementation of TacticZero with GNN encoder and random Induct term selection 

'''
class GNNVanilla(Agent):
    def __init__(self, tactic_pool, replay_dir = None, train_mode = True):
        super().__init__(tactic_pool)

        self.ARG_LEN = 5
        self.train_mode = train_mode
        self.context_rate = 5e-5
        self.tac_rate = 5e-5
        self.arg_rate = 5e-5
        self.term_rate = 5e-5


        self.embedding_dim = 256
        self.gamma = 0.99 # 0.9

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.context_net = utp_model.ContextPolicy().to(self.device)
        self.tac_net = utp_model.TacPolicy(len(tactic_pool)).to(self.device)
        self.arg_net = utp_model.ArgPolicy(len(tactic_pool), self.embedding_dim).to(self.device)
        self.term_net = utp_model.TermPolicy(len(tactic_pool), self.embedding_dim).to(self.device)

        self.induct_gnn = inner_embedding_network.message_passing_gnn_induct(1000, self.embedding_dim // 2, num_iterations=2, device=self.device)
        self.optimizer_induct = torch.optim.RMSprop(list(self.induct_gnn.parameters()), lr = self.term_rate)
        
        self.optimizer_context = torch.optim.RMSprop(list(self.context_net.parameters()), lr=self.context_rate)
        self.optimizer_tac = torch.optim.RMSprop(list(self.tac_net.parameters()), lr=self.tac_rate)
        self.optimizer_arg = torch.optim.RMSprop(list(self.arg_net.parameters()), lr=self.arg_rate)
        self.optimizer_term = torch.optim.RMSprop(list(self.term_net.parameters()), lr=self.term_rate)


        if replay_dir:
            with open(replay_dir) as f:
                self.replays = json.load(f)
        else:
            self.replays = {}

        self.optimizer_encoder_premise = torch.optim.RMSprop(list(self.encoder_premise.parameters()), lr=self.term_rate)
        self.optimizer_encoder_goal = torch.optim.RMSprop(list(self.encoder_goal.parameters()), lr=self.term_rate)

    def load_encoder(self):
        self.encoder_premise = torch.load("/home/sean/Documents/phd/aitp/experiments/hol4/supervised/model_checkpoints/gnn_transformer_premise_hol4")
        self.encoder_goal = torch.load("/home/sean/Documents/phd/aitp/experiments/hol4/supervised/model_checkpoints/gnn_transformer_goal_hol4")
        return

    #todo should be a single line with lightning
    def save(self):
        torch.save(self.context_net, "../model_checkpoints/gnn_induct_context")
        torch.save(self.tac_net, "../model_checkpoints/gnn_induct_tac")
        torch.save(self.arg_net, "../model_checkpoints/gnn_induct_arg")
        torch.save(self.term_net, "../model_checkpoints/gnn_induct_term")
        torch.save(self.induct_gnn, "../model_checkpoints/gnn_induct_gnn")
        torch.save(self.encoder_premise, "../model_checkpoints/gnn_encoder_premise_e2e")
        torch.save(self.encoder_goal, "../model_checkpoints/gnn_encoder_goal_e2e")

    def load(self):
        self.context_net = torch.load("model_checkpoints/gnn_induct_context")
        self.tac_net = torch.load("model_checkpoints/gnn_induct_tac")
        self.arg_net = torch.load("model_checkpoints/gnn_induct_arg")
        self.term_net = torch.load("model_checkpoints/gnn_induct_term")
        self.induct_gnn = torch.load("model_checkpoints/gnn_induct_gnn")

        self.encoder_premise = torch.load("model_checkpoints/gnn_encoder_premise_e2e")
        self.encoder_goal = torch.load("model_checkpoints/gnn_encoder_goal_e2e")

        self.optimizer_context = torch.optim.RMSprop(list(self.context_net.parameters()), lr=self.context_rate)
        self.optimizer_tac = torch.optim.RMSprop(list(self.tac_net.parameters()), lr=self.tac_rate)
        self.optimizer_arg = torch.optim.RMSprop(list(self.arg_net.parameters()), lr=self.arg_rate)
        self.optimizer_term = torch.optim.RMSprop(list(self.term_net.parameters()), lr=self.term_rate)
        self.optimizer_induct = torch.optim.RMSprop(list(self.induct_gnn.parameters()), lr = self.term_rate)
        self.optimizer_encoder_premise = torch.optim.RMSprop(list(self.encoder_premise.parameters()), lr=self.term_rate)
        self.optimizer_encoder_goal = torch.optim.RMSprop(list(self.encoder_goal.parameters()), lr=self.term_rate)

    def run(self, env, allowed_fact_batch, allowed_arguments_ids, candidate_args, max_steps=50):
        
        allowed_fact_batch = allowed_fact_batch.to(self.device)
        # encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
        fringe_pool = []
        tac_pool = []
        arg_pool = []
        action_pool = []
        reward_pool = []
        reward_print = []
        tac_print = []
        induct_arg = []
        proved = 0
        iteration_rewards = []
        steps = 0        
        replay_flag = False

        trace = []
        
        start_t = time.time()

        for t in range(max_steps):
            
            # gather all the goals in the history using goal encoder
            try:
                representations, context_set, fringe_sizes = gather_encoded_content_gnn(env.history, self.encoder_goal)
            except Exception as e:
                print ("Encoder error {}".format(e))
                print (traceback.print_exc())
                return ("Encoder error", str(e))

            encoded_fact_pool = self.encoder_premise(allowed_fact_batch.to(device))

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

            # take the first context in the chosen fringe for now
            try:
                target_context = contexts_by_fringe[fringe][0]
            except:
                print ("error {} {}".format(contexts_by_fringe, fringe))

            target_goal = target_context["polished"]["goal"]
            target_representation = representations[context_set.index(target_context)]

            tac_input = target_representation#.unsqueeze(0)
            tac_input = tac_input.to(self.device)

            # print (tac_input, tac_input.shape)
            tac_probs = self.tac_net(tac_input)
            tac_m = Categorical(tac_probs)
            tac = tac_m.sample()
            tac_pool.append(tac_m.log_prob(tac))
            action_pool.append(tactic_pool[tac])
            tac_print.append(tac_probs.detach())

            tac_tensor = tac.to(self.device)


            if tactic_pool[tac] in no_arg_tactic:
                tactic = tactic_pool[tac]
                arg_probs = []
                arg_probs.append(torch.tensor(0))
                arg_pool.append(arg_probs)
                
            elif tactic_pool[tac] == "Induct_on":

                target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)


                arg_probs = []
                candidates = []

                # tokens = target_goal.split()
                # tokens = list(dict.fromkeys(tokens))
                #

                tokens = [[t] for t in target_graph.labels if t[0] == "V"]

                token_inds = [i for i,t in enumerate(target_graph.labels) if t[0] == "V"]

                if tokens:



                    # pass whole graph through Induct GNN
                    induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)

                    induct_nodes = self.induct_gnn(induct_graph.x.to(self.device), induct_graph.edge_index.to(self.device))



                    # select representations of Variable nodes nodes with ('V' label only)

                    token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))

                    # pass through term_net as before



                    target_representation_list = [target_representation for _ in tokens]

                    target_representations = torch.cat(target_representation_list)

                    candidates = torch.cat([token_representations, target_representations], dim=1)
                    candidates = candidates.to(self.device)


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

                    induct_arg.append(tokens[term])
                    tm = tokens[term][0][1:] # remove headers, e.g., "V" / "C" / ...
                    arg_pool.append(arg_probs)
                    if tm:
                        tactic = "Induct_on `{}`".format(tm)
                        #print (tactic)
                    else:
                        print("tm is empty")
                        print(tokens)
                        # only to raise an error
                        tactic = "Induct_on"
                else:
                    arg_probs.append(torch.tensor(0))
                    induct_arg.append("No variables")
                    arg_pool.append(arg_probs)
                    tactic = "Induct_on"
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
                    return ("hiddenl error...{}", str(e))

                #encode premises with premise GNN
                #encoded_fact_pool = self.encoder_premise.forward(allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                #encode and pool for digae
                #do this at start to avoid recomputation?
                # encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                # encoded_fact_pool = self.encoder_premise(allowed_fact_batch.to(device))
                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(self.device)
                            
                input = tac_tensor
                # print (input.shape, candidates.shape)#, hidden.shape)
                # run it once before predicting the first argument
                hidden, _ = self.arg_net(input, candidates, hidden)

                # the indices of chosen args
                arg_step = []
                arg_step_probs = []
                
                if tactic_pool[tac] in thm_tactic:
                    arg_len = 1
                else:
                    arg_len = self.ARG_LEN#ARG_LEN


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
                    input = encoded_fact_pool[arg].unsqueeze(0)#.unsqueeze(0)

                    # renew candidates                
                    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    hiddenl = torch.cat(hiddenl)
                    #appends both hidden and cell states (when paper only does hidden?)
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(self.device)

                arg_pool.append(arg_step_probs)

                tac = tactic_pool[tac]
                arg = [candidate_args[j] for j in arg_step]


                tactic = env.assemble_tactic(tac, arg)

            action = (fringe.item(), 0, tactic)
            
            trace.append(action)

            #print (action)
            # reward, done = env.step(action)
            try:
                reward, done = env.step(action)

            except:
                print("Step exception raised.")
                return ("Step error", action)
                # print("Fringe: {}".format(env.history))
                print("andling: {}".format(env.handling))
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

            if t == max_steps - 1:
                reward = -5
                
            #could add environment state, but would grow rapidly
            trace.append((reward, action))
            
            reward_print.append(reward)
            reward_pool.append(reward)

            steps += 1
            total_reward = float(np.sum(reward_print))

            if done == True:
                print ("Goal Proved in {} steps".format(t+1))
                iteration_rewards.append(total_reward)

                #if proved, add to successful replays for this goal
                if env.goal in self.replays.keys():
                    #if proof done in less steps than before, add to dict
                    if steps < self.replays[env.goal][0]:
                        print ("adding to replay")
                        print (env.history)
                        self.replays[env.goal] = (steps, env.history)
                else:

                    print ("Initial add to db...")
                    print (env.history)
                    if env.history is not None:
                        self.replays[env.goal] = (steps, env.history)

                    else:
                        print ("history is none.............")
                        print (env.history)
                        print (env)
                break

            if t == max_steps - 1:
                print("Failed.")

                if env.goal in self.replays.keys():
                    replay_flag = True

                    return trace, steps, done, 0, 0, replay_flag

                #print("Rewards: {}".format(reward_print))
                # print("Rewards: {}".format(reward_pool))
                #print("Tactics: {}".format(action_pool))
                # print("Mean reward: {}\n".format(np.mean(reward_pool)))
                #print("Total: {}".format(total_reward))
                iteration_rewards.append(total_reward)


        if self.train_mode:

            self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)
        
        return trace, steps, done, 0, float(np.sum(reward_print)), replay_flag



    def update_params(self, reward_pool, fringe_pool, arg_pool, tac_pool, step_count):
        # Update policy
        # Discount reward
        print("Updating parameters ... ")
        running_add = 0
        for i in reversed(range(step_count)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        self.optimizer_context.zero_grad()
        self.optimizer_tac.zero_grad()
        self.optimizer_arg.zero_grad()
        self.optimizer_term.zero_grad()
        self.optimizer_induct.zero_grad()
        self.optimizer_encoder_premise.zero_grad()
        self.optimizer_encoder_goal.zero_grad()


        total_loss = 0

        for i in range(step_count):
            reward = reward_pool[i]
            
            fringe_loss = -fringe_pool[i] * (reward)
            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            tac_loss = -tac_pool[i] * (reward)
            
            loss = fringe_loss + tac_loss + arg_loss
            total_loss += loss

        total_loss.backward()


        self.optimizer_context.step()
        self.optimizer_tac.step()
        self.optimizer_arg.step()
        self.optimizer_term.step()
        self.optimizer_induct.step()
        self.optimizer_encoder_premise.step()
        self.optimizer_encoder_goal.step()
        return

    def replay_known_proof(self, env, allowed_fact_batch, allowed_arguments_ids, candidate_args):
        #known_history = random.sample(self.replays[env.goal][1], 1)[0]#[0]
        known_history = self.replays[env.goal][1]#[0]

        allowed_fact_batch = allowed_fact_batch.to(self.device)
        fringe_pool = []
        tac_pool = []
        arg_pool = []
        action_pool = []
        reward_pool = []
        reward_print = []
        tac_print = []
        induct_arg = []
        iteration_rewards = []
        steps = 0

        for t in range(len(known_history) - 1):
            true_resulting_fringe = known_history[t + 1]

            try:
                representations, context_set, fringe_sizes = gather_encoded_content_gnn(known_history[:t+1], self.encoder_goal)
            except Exception as e:
                print("Encoder error {}".format(e))
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

            true_fringe = torch.tensor([true_resulting_fringe["parent"]])
            true_fringe = true_fringe.to(self.device)

            fringe_pool.append(fringe_m.log_prob(true_fringe))

            try:
                target_context = contexts_by_fringe[true_fringe][0]
            except:
                print ("error {} {}".format(contexts_by_fringe, true_fringe))

            target_goal = target_context["polished"]["goal"]

            target_representation = representations[context_set.index(target_context)]


            tac_input = target_representation#.unsqueeze(0)
            tac_input = tac_input.to(self.device)

            tac_probs = self.tac_net(tac_input)
            tac_m = Categorical(tac_probs)

            true_tactic_text = true_resulting_fringe["by_tactic"]

            if true_tactic_text in no_arg_tactic:
                true_tac_text = true_tactic_text
            else:
                # true_tactic_text = "Induct_on `ll`"
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


            true_tac = torch.tensor([tactic_pool.index(true_tac_text)])
            true_tac = true_tac.to(self.device)

            tac_pool.append(tac_m.log_prob(true_tac))
            action_pool.append(tactic_pool[true_tac])
            tac_print.append(tac_probs.detach())


            tac_tensor = true_tac.to(self.device)

            assert tactic_pool[true_tac.item()] == true_tac_text




            if tactic_pool[true_tac] in no_arg_tactic:
                tactic = tactic_pool[true_tac]
                arg_probs = []
                arg_probs.append(torch.tensor(0))
                arg_pool.append(arg_probs)


            elif tactic_pool[true_tac] == "Induct_on":
                #TODO need to find a new way to do this...
                #Could pass the whole Graph for the expression then take the embeddings for the specific nodes corresponding to VAR
                #Then pass the embeddings to a new term net to softmax and select one. Need a way to map from VAR in graph back to the variable

                arg_probs = []
                candidates = []

                target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)


                tokens = [[t] for t in target_graph.labels if t[0] == "V"]

                token_inds = [i for i,t in enumerate(target_graph.labels) if t[0] == "V"]

                if tokens:

                    # print ("replaying induction")

                    true_term = torch.tensor([tokens.index(["V" + true_args_text])])

                    true_term = true_term.to(device)

                    #feed through induct GNN to get representation

                    induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)

                    induct_nodes = self.induct_gnn(induct_graph.x.to(self.device), induct_graph.edge_index.to(self.device))

                    # select representations of Variable nodes with ('V' label only)

                    token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))

                    # pass through term_net as before



                    target_representation_list = [target_representation for _ in tokens]

                    target_representations = torch.cat(target_representation_list)

                    candidates = torch.cat([token_representations, target_representations], dim=1)
                    candidates = candidates.to(self.device)


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

                    arg_probs.append(term_m.log_prob(true_term))

                    induct_arg.append(tokens[true_term])
                    tm = tokens[true_term][0][1:] # remove headers, e.g., "V" / "C" / ...
                    arg_pool.append(arg_probs)

                    assert tm == true_args_text

                    if tm:
                        tactic = "Induct_on `{}`".format(tm)
                    else:
                        print("tm is empty")
                        print(tokens)
                        # only to raise an error
                        tactic = "Induct_on"
                else:
                    arg_probs.append(torch.tensor(0))
                    induct_arg.append("No variables")
                    arg_pool.append(arg_probs)
                    tactic = "Induct_on"



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
                    return ("hiddenl error...{}", str(e))


                # encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device),allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                encoded_fact_pool = self.encoder_premise(allowed_fact_batch.to(device))
                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(self.device)

                input = tac_tensor

                # run it once before predicting the first argument
                hidden, _ = self.arg_net(input, candidates, hidden)

                # the indices of chosen args
                arg_step = []
                arg_step_probs = []

                if tactic_pool[true_tac] in thm_tactic:
                    arg_len = 1
                else:
                    arg_len = self.ARG_LEN#ARG_LEN


                for i in range(arg_len):
                    hidden, scores = self.arg_net(input, candidates, hidden)
                    arg_probs = F.softmax(scores, dim=0)
                    arg_m = Categorical(arg_probs.squeeze(1))

                    if isinstance(true_args_text, list):
                        try:
                            name_parser = true_args_text[i].split(".")
                        except:
                            print (i)
                            print (true_args_text)
                            print (known_history)
                            exit()
                        theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                        theorem_name = name_parser[1]
                        #todo not sure if reverse_database will work...
                        true_arg_exp = reverse_database[(theory_name, theorem_name)]
                    else:
                        name_parser = true_args_text.split(".")
                        theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                        theorem_name = name_parser[1]
                        true_arg_exp = reverse_database[(theory_name, theorem_name)]
                    true_arg = torch.tensor(candidate_args.index(true_arg_exp))
                    true_arg = true_arg.to(self.device)



                    arg_step.append(true_arg)
                    arg_step_probs.append(arg_m.log_prob(true_arg))

                    hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
                    hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

                    # encoded chosen argument
                    input = encoded_fact_pool[true_arg].unsqueeze(0)#.unsqueeze(0)

                    # renew candidates
                    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    hiddenl = torch.cat(hiddenl)
                    #appends both hidden and cell states (when paper only does hidden?)
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(self.device)

                arg_pool.append(arg_step_probs)
            try:
                reward = true_resulting_fringe["reward"]
                #done = t + 2 == len(known_history)
            except:
                print("Step exception raised.")
                return ("Step error", action)
                # print("Fringe: {}".format(env.history))
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

            reward_print.append(reward)
            reward_pool.append(reward)
            steps += 1

        total_reward = float(np.sum(reward_print))

        if self.train_mode:
            self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)

        return

    def save_replays(self):
        with open("old_replays/gnn_induct_agent_replays.json", "w") as f:
            json.dump(self.replays, f)


class Experiment_GNN:
    def __init__(self, agent, goals, database, num_iterations, train_mode= True):
        self.agent = agent
        self.train_mode = train_mode
        self.goals = goals
        self.num_iterations = num_iterations
        self.database = database

    def train(self):
        env = HolEnv("T")
        env_errors = []
        agent_errors = []
        full_trace = []
        iter_times = []
        reward_trace = []
        proved_trace = []

        for iteration in range(self.num_iterations):
            iter_trace = {}
            it_start = time.time()

            prove_count = 0

            for i, goal in enumerate(self.goals):
                print ("Goal #{}".format(str(i+1)))

                try:
                    env.reset(goal[1])
                except Exception as e:
                    print ("Restarting environment..")
                    env = HolEnv("T")
                    #env_errors.append((goal, e, i))
                    continue
                 
                try:
                    allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(env, goal)
                except Exception as e:
                    print ("Env error: {}".format(e))
                    #env_errors.append((goal, "Error generating fact pool", i))
                    print (traceback.print_exc())
                    continue
                    
                result = self.agent.run(env, 
                                        allowed_fact_batch,
                                        allowed_arguments_ids, candidate_args,  max_steps=50)

                
                #agent run returns (error_msg, details) if error, larger tuple otherwise
                if len(result) == 2:
                    #agent_errors.append((result, i))
                    pass

                else:
                    trace, steps, done, goal_time, reward_total, replay_flag = result
                    #iter_trace[goal[0]] = (trace, steps, done, goal_time, reward_total)



                    if replay_flag:
                        print ("replaying proof...")
                        #todo reset env?
                        #reset environment before replay
                        try:
                            env.reset(goal[1])
                        except Exception as e:
                            print ("Restarting environment..")
                            env = HolEnv("T")
                            continue

                        print ("env reset...")

                        try:
                            self.agent.replay_known_proof(env, allowed_fact_batch, allowed_arguments_ids, candidate_args)
                        except Exception as e:
                            print (f"replay error {e}")


                    reward_trace.append(reward_total)
                    if done:
                        prove_count += 1

            if self.train_mode:
                self.agent.save_replays()
            #full_trace.append(iter_trace)
            #iter_times.append(time.time() - it_start)
            proved_trace.append(prove_count)

            if self.train_mode:
                date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')


                # with open(f"traces/gnn_induction_agent_reward_trace_{date}.pk", "wb") as f:
                #     pickle.dump(reward_trace, f)
                #
                # with open("traces/gnn_model_induction_errors.pk", "wb") as f:
                #     pickle.dump((env_errors, agent_errors), f)
                #
                # with open(f"traces/gnn_model_induction_proved_{date}.pk", "wb") as f:
                #     pickle.dump(proved_trace, f)


                #save parameters every iteration
                # self.agent.save()

            print (prove_count)
            print ("wefwef")
        return #full_trace, env_errors, agent_errors, iter_times
            
                
                
    def load_encoded_db(self, encoded_db_dir):
        self.encoded_database = torch.load(encoded_db_dir)

    def load_db(self, db_dir):
        with open(db_dir) as f:
            self.database = json.load(f)

    def gen_fact_pool(self, env, goal):

        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', goal[0])))
        goal_theory = self.database[goal[0]][0]

        polished_goal = env.fringe["content"][0]["polished"]["goal"]

        try:
            allowed_arguments_ids = []
            candidate_args = []
            for i,t in enumerate(self.database):
                # if self.database[t][0] in allowed_theories and (self.database[t][0] != goal_theory or int(self.database[t][2]) < int(self.database[polished_goal][2])):
                if self.database[t][0] in allowed_theories and (self.database[t][0] != goal_theory or int(self.database[t][3]) < int(self.database[polished_goal][3])):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)

            env.toggle_simpset("diminish", goal_theory)
            print("Removed simpset of {}".format(goal_theory))

        except:
            allowed_arguments_ids = []
            # candidate_args = []
            # for i,t in enumerate(self.database):
            #     if self.database[t][0] in allowed_theories:
            #         allowed_arguments_ids.append(i)
            #         candidate_args.append(t)
            raise Exception("Theorem not found in database.")


        graphs = [graph_db[t] for t in candidate_args]
        
        loader = DataLoader(graphs, batch_size = len(candidate_args))

        
        allowed_fact_batch = next(iter(loader))


        allowed_fact_batch.edge_attr = allowed_fact_batch.edge_attr.long()
        # encoded_fact_pool = gnn_enc.forward(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))

        # return encoded_fact_pool, allowed_arguments_ids, candidate_args
        #return batch with graphs to encode by agent
        return allowed_fact_batch, allowed_arguments_ids, candidate_args




def run_experiment():
    try:
        agent = GNNVanilla(tactic_pool)#, replay_dir="old_replays/gnn_induct_agent_replays.json")

        # agent.load()

        exp_gnn = Experiment_GNN(agent, train_goals, compat_db, 800)

        exp_gnn.train()

    except Exception as e:
        print (f"Fatal error {e}")
        print (traceback.print_exc())
        run_experiment()

def run_test():
    try:
        agent = GNNVanilla(tactic_pool, replay_dir=None, train_mode=False)

        # agent.load()

        exp_gnn = Experiment_GNN(agent, test_goals, compat_db, 1, train_mode=False)

        exp_gnn.train()

    except Exception as e:
        print (f"Fatal error {e}")
        print (traceback.print_exc())
        # run_experiment()




# run_experiment()
run_test()


#sanity check encodings are similar between non-deterministic runs
# cs = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

# x = []
# for i,k in enumerate(compat_db.keys()):
#     test = torch_graph_dict[k]
#     x.append(cs(gnn_enc.forward(test.x.to(device), test.edge_index.to(device)), encoded_graph_db[i]).cpu().detach().numpy())
#     #assert gnn_enc.forward(test.x.to(device), test.edge_index.to(device))[0][0] == encoded_graph_db[i][0][0]

# np.mean(x)

