# import traceback
# from collections import OrderedDict, deque, namedtuple
# import lightning.pytorch as pl
# from data.hol4.ast_def import graph_to_torch_labelled
# from torch.distributions import Categorical
# import torch.nn.functional as F
# from datetime import datetime
# import pickle
# from data.hol4 import ast_def
# from torch_geometric.loader import DataLoader
# from models.tactic_zero import policy_models
# from models.gnn.formula_net import formula_net
# import time
# from environments.hol4.new_env import *
# import numpy as np
# #import batch_gnn
#
#
# # todo first try with fringes and replays as in original, then can make class for proper replay buffer with graph states
# class TacticZeroGNN(pl.LightningModule):
#     def __init__(self,
#                  goal_net,
#                  tac_net,
#                  arg_net,
#                  batch_size=32,
#                  lr=1e-4):
#
#         super().__init__()
#
#         self.goal_net = goal_net
#         self.tac_net = tac_net
#         self.arg_net = arg_net
#
#         self.lr = lr
#         self.batch_size = batch_size
#
#         self.save_hyperparameters()
#
#     def forward(self, goal, premise):
#         # have a method for a single run through or for a batch from replay buffer
#         # e.g. for online rewards, probs = self.run_proof
#         # loss = calc_loss(rewards, probs)
#
#
#
#     def training_step(self, batch, batch_idx):
#     def validation_step(self, batch, batch_idx):
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
#         return optimizer
#
#
# Trajectory = namedtuple(
#     "Trajectory",
#     field_names=["history", "action_history", "reward", "new_state"],
# )
#
#
# class Agent:
#     def __init__(self, env, replay_buffer):
#         self.env = env
#         self.replay_buffer = replay_buffer
#         self.reset()
#
#     def get_action(self, ):
#
#         ######################################################################################################
#         # first need an abstract interface to get goal from state
#         ######################################################################################################
#
#         try:
#             representations, context_set, fringe_sizes = gather_encoded_content_gnn(env.history, self.encoder_goal)
#         except Exception as e:
#             print("Encoder error {}".format(e))
#             print(traceback.print_exc())
#             return ("Encoder error", str(e))
#
#         encoded_fact_pool = self.encoder_premise(allowed_fact_batch.to(device))
#
#         context_scores = self.context_net(representations)
#         contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
#         fringe_scores = []
#
#         for s in scores_by_fringe:
#             fringe_score = torch.sum(s)
#             fringe_scores.append(fringe_score)
#         fringe_scores = torch.stack(fringe_scores)
#
#         fringe_probs = F.softmax(fringe_scores, dim=0)
#         fringe_m = Categorical(fringe_probs)
#         fringe = fringe_m.sample()
#         fringe_pool.append(fringe_m.log_prob(fringe))
#
#         # take the first context in the chosen fringe for now
#         try:
#             target_context = contexts_by_fringe[fringe][0]
#         except:
#             print("error {} {}".format(contexts_by_fringe, fringe))
#
#         target_goal = target_context["polished"]["goal"]
#         target_representation = representations[context_set.index(target_context)]
#
#
#         ######################################################################################################
#         #  get tactic
#         ######################################################################################################
#
#
#         tac_input = target_representation  # .unsqueeze(0)
#         tac_input = tac_input.to(self.device)
#
#         # print (tac_input, tac_input.shape)
#         tac_probs = self.tac_net(tac_input)
#         tac_m = Categorical(tac_probs)
#         tac = tac_m.sample()
#         tac_pool.append(tac_m.log_prob(tac))
#         action_pool.append(tactic_pool[tac])
#         tac_print.append(tac_probs.detach())
#
#         tac_tensor = tac.to(self.device)
#
#         ######################################################################################################
#         #  get args
#         ######################################################################################################
#
#         if tactic_pool[tac] in no_arg_tactic:
#             tactic = tactic_pool[tac]
#             arg_probs = []
#             arg_probs.append(torch.tensor(0))
#             arg_pool.append(arg_probs)
#
#         elif tactic_pool[tac] == "Induct_on":
#
#             target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)
#
#             arg_probs = []
#             candidates = []
#
#             # tokens = target_goal.split()
#             # tokens = list(dict.fromkeys(tokens))
#             #
#
#             tokens = [[t] for t in target_graph.labels if t[0] == "V"]
#
#             token_inds = [i for i, t in enumerate(target_graph.labels) if t[0] == "V"]
#
#             if tokens:
#
#                 # pass whole graph through Induct GNN
#                 induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)
#
#                 induct_nodes = self.induct_gnn(induct_graph.x.to(self.device), induct_graph.edge_index.to(self.device))
#
#                 # select representations of Variable nodes nodes with ('V' label only)
#
#                 token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))
#
#                 # pass through term_net as before
#
#                 target_representation_list = [target_representation for _ in tokens]
#
#                 target_representations = torch.cat(target_representation_list)
#
#                 candidates = torch.cat([token_representations, target_representations], dim=1)
#                 candidates = candidates.to(self.device)
#
#                 scores = self.term_net(candidates, tac_tensor)
#                 term_probs = F.softmax(scores, dim=0)
#                 try:
#                     term_m = Categorical(term_probs.squeeze(1))
#                 except:
#                     print("probs: {}".format(term_probs))
#                     print("candidates: {}".format(candidates.shape))
#                     print("scores: {}".format(scores))
#                     print("tokens: {}".format(tokens))
#                     exit()
#
#                 term = term_m.sample()
#
#                 arg_probs.append(term_m.log_prob(term))
#
#                 induct_arg.append(tokens[term])
#                 tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
#                 arg_pool.append(arg_probs)
#                 if tm:
#                     tactic = "Induct_on `{}`".format(tm)
#                     # print (tactic)
#                 else:
#                     print("tm is empty")
#                     print(tokens)
#                     # only to raise an error
#                     tactic = "Induct_on"
#             else:
#                 arg_probs.append(torch.tensor(0))
#                 induct_arg.append("No variables")
#                 arg_pool.append(arg_probs)
#                 tactic = "Induct_on"
#         else:
#             hidden0 = hidden1 = target_representation
#
#             hidden0 = hidden0.to(self.device)
#             hidden1 = hidden1.to(self.device)
#
#             hidden = (hidden0, hidden1)
#
#             # concatenate the candidates with hidden states.
#
#             hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
#             hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]
#
#             try:
#                 hiddenl = torch.cat(hiddenl)
#             except Exception as e:
#                 return ("hiddenl error...{}", str(e))
#
#             # encode premises with premise GNN
#             # encoded_fact_pool = self.encoder_premise.forward(allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
#             # encode and pool for digae
#             # do this at start to avoid recomputation?
#             # encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
#             # encoded_fact_pool = self.encoder_premise(allowed_fact_batch.to(device))
#             candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
#             candidates = candidates.to(self.device)
#
#             input = tac_tensor
#             # print (input.shape, candidates.shape)#, hidden.shape)
#             # run it once before predicting the first argument
#             hidden, _ = self.arg_net(input, candidates, hidden)
#
#             # the indices of chosen args
#             arg_step = []
#             arg_step_probs = []
#
#             if tactic_pool[tac] in thm_tactic:
#                 arg_len = 1
#             else:
#                 arg_len = self.ARG_LEN  # ARG_LEN
#
#             for _ in range(arg_len):
#                 hidden, scores = self.arg_net(input, candidates, hidden)
#                 arg_probs = F.softmax(scores, dim=0)
#                 arg_m = Categorical(arg_probs.squeeze(1))
#                 arg = arg_m.sample()
#                 arg_step.append(arg)
#                 arg_step_probs.append(arg_m.log_prob(arg))
#
#                 hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
#                 hidden1 = hidden[1].squeeze().repeat(1, 1, 1)
#
#                 # encoded chosen argument
#                 input = encoded_fact_pool[arg].unsqueeze(0)  # .unsqueeze(0)
#
#                 # renew candidates
#                 hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
#                 hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]
#
#                 hiddenl = torch.cat(hiddenl)
#                 # appends both hidden and cell states (when paper only does hidden?)
#                 candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
#                 candidates = candidates.to(self.device)
#
#             arg_pool.append(arg_step_probs)
#
#             tac = tactic_pool[tac]
#             arg = [candidate_args[j] for j in arg_step]
#
#             tactic = env.assemble_tactic(tac, arg)
#
#         action = (fringe.item(), 0, tactic)
#         # trace.append(action)
#         # print (action)
#         # reward, done = env.step(action)
#
#         #####################################################################################################
#         #####################################################################################################
#         #####################################################################################################
#
#     # Named tuple for storing experience steps gathered in training
#
#     # history will be all states, actions, rewards and model probabilities for e.g. importance sampling.
#
#
#     def play_step(self):
#         action = self.get_action()
#
#         try:
#             reward, done = self.env.step(action)
#         except:
#             print("Step exception raised.")
#             return ("Step error", action)
#             # print("Fringe: {}".format(env.history))
#             print("andling: {}".format(env.handling))
#             print("Using: {}".format(env.using))
#             # try again
#             # counter = env.counter
#             frequency = env.frequency
#             env.close()
#             print("Aborting current game ...")
#             print("Restarting environment ...")
#             print(env.goal)
#             env = HolEnv(env.goal)
#             flag = False
#             break
#
#
#         if done:
#             traj = Trajectory(self.env.history, self.env.action_history, reward_pool,
#             (torch.stack(fringe_pool).cpu().detach().numpy(), torch.stack(tac_pool).cpu().detach().numpy(),
#             [torch.stack(arg).cpu().detach().numpy() for arg in arg_pool])))
#
#
#             self.replay_buffer[self.goal].append(traj)
#
#             self.reset()
#
#         return reward, done
#
#         if t == max_steps - 1:
#             reward = -5
#
#         # could add environment state, but would grow rapidly
#         trace.append((reward, action))
#
#         reward_print.append(reward)
#         reward_pool.append(reward)
#
#         steps += 1
#         total_reward = float(np.sum(reward_print))
#
#         if done == True:
#             print("Goal Proved in {} steps".format(t + 1))
#             iteration_rewards.append(total_reward)
#
#             # if proved, add to successful replays for this goal
#             # todo need data module for replays. MongoDB can hold the replays with goal as key. Can then generate
#             # todo HDF5 dataset on demand, with e.g. prioritised batch of trajectories with goal state, expressions, premises, log probs etc. precomputed for training
#
#
#