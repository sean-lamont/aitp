import os
import sys

from environments.int_environment.algos.lib.obs import one_hot, thm2index, index2thm, theorem_no_input, thm_index2no_input, convert_obs_to_dict, \
    convert_batch_obs_to_dict

sys.path.insert(0, os.path.abspath('../../Inequality'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.int_environment.algos.gnns import FCResBlock
from environments.int_environment.algos.transformer.modules.models import make_thm_model
from torch_geometric.data import Batch
from torch.distributions import Categorical, Uniform
from torch_scatter import scatter_add, scatter_max
import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# device = torch.device("cpu")
# cuda = False


class GroundTruthEncoderTransformer(torch.nn.Module):
    def __init__(self, num_in, num_out, nn_dim=64, gnn_type="GAT", hidden_layers=1):
        super(GroundTruthEncoderTransformer, self).__init__()
        self.gnn_type = gnn_type
        self.h_size = nn_dim
        self.n_in = num_in
        self.graph_encoder = make_thm_model(num_in=num_in, N=6,
                                            dim_model=nn_dim, dim_ff=2048, h=8, dropout=0.1)
        self.to(device)

    def forward(self, data, gnn_ind):

        # TODO: attention?
        state_tensor = self.graph_encoder(data)
        out = scatter_add(state_tensor, gnn_ind, 0)
        import pdb;pdb.set_trace()
        return state_tensor, out


class ThmNet(torch.nn.Module):

    def __init__(self, **options):
        super().__init__()
        self.device = device
        num_nodes = options["num_nodes"]
        num_lemmas = options["num_lemmas"]
        state_dim = options["state_dim"]
        gnn_type = options["gnn_type"]
        combined_gt_obj = options["combined_gt_obj"]
        attention_type = options["attention_type"]
        hidden_layers = options["hidden_layers"]
        self.entity_cost = options["entity_cost"]
        self.lemma_cost= options["lemma_cost"]
        if options["pretrain"] is not None:
            pretrain = options["pretrain"]
        else:
            pretrain = False
        norm = options["norm"]

        if pretrain:
            checkpoint = torch.load(str(pretrain)+"model_checkpoint.pt", map_location='cpu')
            model = checkpoint["model"]
            self.encoder = model.encoder
            self.lemma_encoder = model.lemma_encoder
            self.key_transform = model.key_transform
            self.ent_transform = model.ent_transform
            self.gt_ent_transform = model.gt_ent_transform
            self.obj_ent_transform = model.obj_ent_transform
            self.lemma_q = model.lemma_q
            self.vf_net = model.vf_net
            self.entity_q = model.entity_q
            self.combined_gt_obj = model.combined_gt_obj
            self.attention_type = model.attention_type
        else:
            self.encoder = GroundTruthEncoderTransformer(num_nodes + 4, state_dim, state_dim,
                                                            gnn_type=gnn_type, hidden_layers=hidden_layers)
            self.lemma_encoder = nn.Linear(num_lemmas, 2*state_dim)
            self.key_transform = nn.Linear(2*state_dim, state_dim, bias=False)
            self.ent_transform = nn.Linear(state_dim, 2*state_dim, bias=False)
            self.gt_ent_transform = nn.Linear(state_dim, state_dim, bias=False)
            self.obj_ent_transform = nn.Linear(state_dim, state_dim, bias=False)
            # self.lemma_q = nn.Sequential(
            #     nn.Linear(2*state_dim, state_dim),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(state_dim, num_lemmas))
            self.lemma_q = FCResBlock(2*state_dim)
            self.lemma_out = nn.Linear(2*state_dim, num_lemmas)
            self.vf_net = nn.Sequential(
                nn.Linear(2 * state_dim, state_dim),
                nn.ReLU(inplace=True),
                nn.Linear(state_dim, 1))
            self.entity_q = nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(inplace=True),
                nn.Linear(state_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.to(device)

    def vf(self, obs):
        obs = convert_obs_to_dict(obs)
        obj_state = obs["obj"]
        gt_state = obs["gt"]
        obj_gnn = dgl.batch(obj_state["obj_graph"])

        obj_state_tensor, obj_out = self.obj_encoder.forward(obj_gnn, obj_state["obj_gnn_ind"])

        if len(gt_state["gt_graph"]) > 0:
            gt_gnn = dgl.batch(gt_state["gt_graph"])
            gt_state_tensor, gt_out = self.gt_encoder.forward(gt_gnn, gt_state["gt_gnn_ind"])
        else:
            gt_out = torch.zeros_like(obj_out)


        out = torch.cat((obj_out, gt_out), 1)

        vf = self.vf_net(out)
        return vf

    def compute_action(self, obs, sample=True):
        obs = convert_obs_to_dict(obs)
        obj_state = obs["obj"]
        gt_state = obs["gt"]
        node_ent = obs["node_ent"]
        obj_gnn = dgl.batch(obj_state["obj_graph"])
        obj_state_tensor, obj_out = self.obj_encoder.forward(obj_gnn, obj_state["obj_gnn_ind"])
        if len(gt_state["gt_graph"]) > 0:
            gt_gnn = dgl.batch(gt_state["gt_graph"])
            gt_state_tensor, gt_out = self.gt_encoder.forward(gt_gnn, gt_state["gt_gnn_ind"])
            state_tensor = torch.cat((obj_state_tensor, gt_state_tensor), 0)
        else:
            gt_out = torch.zeros_like(obj_out)
            state_tensor = obj_state_tensor
        out = torch.cat((obj_out, gt_out), 1)

        vf = self.vf_net(out)

        # TODO: A modular q net for lemma?
        # lemma_outputs = self.lemma_q(out)
        lemma_outputs = self.lemma_out(F.relu(self.lemma_q(out)))
        if sample:
            lemma_m = Categorical(probs=self.softmax(lemma_outputs))
            lemma_action = lemma_m.sample()
            lemma_action = lemma_action.squeeze()
        else:
            lemma_action = torch.argmax(lemma_outputs)
        batch_lemmas = one_hot(lemma_action, len(thm2index))
        # TODO: add nonlinearities when merging info?
        out = F.relu(out + F.relu(self.lemma_encoder(batch_lemmas)))

        assert len(node_ent) == len(obj_state["obj_gnn_ind"]) + len(gt_state["gt_gnn_ind"])
        # select node that is not a logic formula
        if cuda:
            node_ent = torch.cuda.LongTensor(node_ent)
        else:
            node_ent = torch.LongTensor(node_ent)

        ent_mask = (node_ent.cpu() != -1)
        ent_rep = state_tensor[ent_mask, :].clone()

        # TODO: Check numerical issues from torch tensor to numpy
        lemma_action = lemma_action.cpu().numpy().tolist()
        if isinstance(lemma_action, list):
            lemma_action = lemma_action[0]
        num_ent = thm_index2no_input[lemma_action]
        actions = [lemma_action]
        # TODO: fix this:
        for ii in range(num_ent):
            # get the key for each entity
            key = out
            h = torch.mean(self.key_transform(key) * ent_rep, 1)
            entity_m = Categorical(logits=h)
            entity_action = entity_m.sample()
            # print("ent action:", entity_action)
            # TODO: add nonlinearities when merging info?
            out = F.relu(out + F.relu(self.ent_transform(torch.index_select(ent_rep, 0, entity_action))))
            # add one because 0 is saved for NO_OP
            actions.append(entity_action.squeeze().cpu().numpy() + 1)

        while len(actions) < 6:
            actions.append(0)
        # action = torch.LongTensor(actions)
        action = np.array(actions)
        # action = np.array(actions)[np.newaxis, :]
        return action, vf.squeeze(1)

    def batch_forward(self, observations, actions, name_actions=None, acc=False, rl_batch=False):
        gnns = []
        node_name = []
        node_ent = []
        batch_ind = []
        gnn_ind = []
        prev_max_ent = 0
        # if isinstance(observations, np.ndarray) or isinstance(observations, list):
        if rl_batch:
            flatten_obs = []
            for i in range(len(observations)):
                flatten_obs.extend(observations[i])
            actions = np.vstack([ac for ac in actions])
            observations = flatten_obs
            observations = convert_batch_obs_to_dict(observations)
        entity_actions = actions[:, 1:] - 1 # recover entity indices

        for ii, obs in enumerate(observations):
            gnns.append(obs["g"])
            node_name += obs["node_name"]
            node_ent += obs["node_ent"]
            num_ent = sum(np.array(obs["node_ent"])!=-1)
            # shift index for entity actions
            entity_actions[ii] = entity_actions[ii] + prev_max_ent * (entity_actions[ii] != -1).astype(float)
            prev_max_ent += num_ent
            # node index -> problem index
            gnn_ind += [ii] * len(obs["node_ent"])
            # entity index -> problem index
            batch_ind += [ii] * num_ent
            # print(sum(np.array(obs[2])!=-1))
        if cuda:
            batch_ind = torch.cuda.LongTensor(batch_ind)
            gnn_ind = torch.cuda.LongTensor(gnn_ind)
        else:
            batch_ind = torch.LongTensor(batch_ind)
            gnn_ind = torch.LongTensor(gnn_ind)
        batch_state = dgl.batch(gnns)
        state_tensor, out = self.encoder(batch_state, gnn_ind)
        vf = self.vf_net(out)

        lemma_outputs = self.lemma_out(F.relu(self.lemma_q(out)))
        lemma_m = Categorical(probs=self.softmax(lemma_outputs))

        if cuda:
            lemma_actions = torch.cuda.LongTensor(actions[:, 0])
        else:
            lemma_actions = torch.LongTensor(actions[:, 0])

        lemma_logprobs = lemma_m.log_prob(lemma_actions)
        # lemma_logprobs = (lemma_logprobs * (lemma_actions == 14).float() * 1 / 10 +
        #                   lemma_logprobs * (1 - (lemma_actions == 14).float()))
        lemma_entropy = torch.mean(lemma_m.entropy())
        if acc:
            lemma_acc = torch.mean((lemma_outputs.argmax(1)==lemma_actions).float())
            print(lemma_outputs.argmax(1).cpu().numpy())
            print(lemma_actions)
            print("lemma_acc: {}".format(lemma_acc.cpu().numpy()))

            different_lemma_indices = []
            if lemma_acc < 1:
                different_obs_indices = (lemma_outputs.argmax(1).cpu() != lemma_actions.cpu()).nonzero().numpy().tolist()

                different_obs_indices = [inner_list[0] for inner_list in different_obs_indices]

                different_lemma_indices = [
                    {
                        "Observation index": ind,
                        "Label lemma index": lemma_actions.cpu()[ind],
                        "Chosen lemma index": lemma_outputs.argmax(1).cpu()[ind]
                    } for ind in different_obs_indices
                ]

        batch_lemmas = one_hot(lemma_actions, len(thm2index))
        # TODO: add nonlinearities when merging info?
        out = F.relu(out.to(device) + F.relu(self.lemma_encoder(batch_lemmas).to(device)))

        if cuda:
            node_ent = torch.LongTensor(node_ent).cuda()
            ent_mask = (node_ent != -1).cuda()
            node_name = torch.LongTensor(node_name).cuda()
        else:
            node_ent = torch.LongTensor(node_ent)
            ent_mask = (node_ent != -1)
            node_name = torch.LongTensor(node_name)
        ent_rep = state_tensor[ent_mask, :]

        max_num_ent = 4
        ent_logprob = 0.
        ent_entropy = 0.
        if acc:
            num_ent_correct = 0
            num_name_correct = 0
            num_ent = 0
        if cuda:
            entity_actions = torch.cuda.LongTensor(entity_actions)
        else:
            entity_actions = torch.LongTensor(entity_actions)
        if name_actions is not None:
            name_actions = torch.LongTensor(name_actions)
        # TODO: Check numerical issues from torch tensor to numpy
        for ii in range(max_num_ent):
            if cuda:
                mask = (entity_actions[:, ii] != -1).cuda()
            else:
                mask = (entity_actions[:, ii] != -1)
            if sum(mask) > 0:
                key = torch.index_select(out, 0, batch_ind)
                h = torch.mean(self.key_transform(key) * ent_rep, 1)
                # calculate loss term.
                max_h = torch.index_select(
                    scatter_max(h, batch_ind)[0], 0, batch_ind)
                logsoftmax = h - max_h - torch.index_select(
                    torch.log(scatter_add(torch.exp(h-max_h), batch_ind)), 0, batch_ind)
                # mask out unused loss.
                ent_logprob = ent_logprob + torch.mean(torch.index_select(
                    logsoftmax, 0, entity_actions[:, ii][mask].to(device)))
                zero_tensor = torch.zeros(entity_actions.shape[0]).to(device)
                zero_ind = torch.LongTensor(np.arange(entity_actions.shape[0])).to(device)
                cur_logsoftmax = torch.index_select(
                    logsoftmax, 0, entity_actions[:, ii][mask].to(device))
                cur_logsoftmax = \
                    scatter_add(torch.cat((cur_logsoftmax, zero_tensor),0),
                                torch.cat(
                                    (torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(device),
                                     zero_ind), 0))
                ent_logprob = ent_logprob + cur_logsoftmax

                # calculate entropy.
                prob = h / torch.index_select(
                    scatter_add(torch.exp(h), batch_ind), 0, batch_ind)
                # mask out unused probs.
                # TODO: Check whether scale makes sense? sum entropy of different entity action? but mean over data?
                ent_entropy = ent_entropy + torch.mean(scatter_add(logsoftmax * prob, batch_ind)[mask])
                cur_ent_rep = torch.index_select(
                    ent_rep, 0, entity_actions[:, ii][mask].to(device))
                zero_tensor = torch.zeros(entity_actions.shape[0], cur_ent_rep.shape[1]).to(device)
                cur_ent_rep = \
                    scatter_add(torch.cat((cur_ent_rep, zero_tensor), 0),
                                torch.cat(
                                    (torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(device),
                                     zero_ind), 0), 0)
                if acc:
                    h_max = scatter_max(h, batch_ind)[1].to(device)
                    num_ent += mask.float().sum()
                    num_ent_correct += (entity_actions[:, ii].to(device) == h_max).float()[mask].sum()
                    num_name_correct += (node_name[entity_actions[:, ii].to(device)] == node_name[h_max]).float()[
                        mask].sum()
                    # print(node_name[rev_trans_ind][entity_actions[:, ii]])
                    # if name_actions is not None:
                    #     print(name_actions[:, ii])
                    # num_name_correct += (name_actions[:, ii].to(device)==node_name[h_max]).float()[mask].sum()
                    # print("Ground truth entity indices:\n", entity_actions[:, ii].cpu().numpy())
                    # print("Prediction entity indices:\n", h_max.cpu().numpy())
                    # print("Matched entries accumulated: ", num_ent_correct.cpu().numpy())
                    # print("Ground truth name indices:\n", name_actions[:, ii].cpu().numpy())
                    # print("Prediction name indices:\n", node_name[h_max].cpu().numpy())
                    # print("Matched entries accumulated: ", num_name_correct.cpu().numpy())

                out = F.relu(out + F.relu(self.ent_transform(cur_ent_rep)))
        logprobs = self.lemma_cost * lemma_logprobs + self.entity_cost * ent_logprob
        entropy = lemma_entropy #+ ent_entropy
        # action = torch.LongTensor(actions)
        if acc:
            ent_acc = num_ent_correct / num_ent
            name_acc = num_name_correct / num_ent
            print("ent_acc:{}, name_acc:{}".format(ent_acc.cpu().numpy(), name_acc.cpu().numpy()))
            assert name_acc >= ent_acc
            acc = (lemma_acc, ent_acc, name_acc, different_lemma_indices)
            # acc = (lemma_acc, torch.zeros(1), torch.zeros(1), different_lemma_indices)
            return logprobs, vf.squeeze(1), entropy, acc

        return logprobs, vf.squeeze(1), entropy

    def batch_compute_action(self, observations):
        all_gnns = []
        obj_gnns = []
        obj_gnns_ind = []
        obj_batch_gnn_ind = []
        obj_prev_num_graphs = 0
        obj_node_ent = []
        obj_node_name = []
        gt_gnns = []
        gt_gnns_ind = []
        gt_batch_gnn_ind = []
        gt_prev_num_graphs = 0
        gt_node_ent = []
        gt_node_name = []
        batch_ind = []
        prev_max_ent = 0
        prev_max_ents = []


        obj_trans_ind = []
        gt_trans_ind = []
        if isinstance(observations, np.ndarray) or isinstance(observations, list):
            observations = convert_batch_obs_to_dict(observations)

        for ii, obs in enumerate(observations):
            obj_gnn = obs["obj"]["obj_graph"]
            obj_gnn_ind = obs["obj"]["obj_gnn_ind"]
            # node index -> graph index
            obj_gnns_ind += [obj_prev_num_graphs+ind for ind in obj_gnn_ind]
            obj_prev_num_graphs = max(obj_gnns_ind) + 1
            obj_batch_gnn_ind += [ii] * (max(obj_gnn_ind) + 1)
            # graph index -> problem index
            obj_gnns += obj_gnn
            all_gnns += obj_gnn
            if len(obs["gt"]["gt_graph"]) > 0:
                gt_gnn = obs["gt"]["gt_graph"]
                gt_gnn_ind = obs["gt"]["gt_gnn_ind"]
                gt_gnns_ind += [gt_prev_num_graphs+ind for ind in gt_gnn_ind]
                gt_prev_num_graphs = max(gt_gnns_ind) + 1
                gt_batch_gnn_ind += [ii] * (max(gt_gnn_ind) + 1)
                all_gnns += gt_gnn
                gt_gnns += gt_gnn
            else:
                gt_gnn_ind = []

            assert len(obs["node_ent"]) == len(obj_gnn_ind) + len(gt_gnn_ind)
            obj_node_ent += obs["node_ent"][:len(obj_gnn_ind)]
            obj_node_name += obs["node_name"][:len(obj_gnn_ind)]
            gt_node_ent += obs["node_ent"][len(obj_gnn_ind):]
            gt_node_name += obs["node_name"][len(obj_gnn_ind):]

            num_ent = sum(np.array(obs["node_ent"])!=-1)
            num_obj_ent = len(np.array(obs["node_ent"][:len(obj_gnn_ind)])[np.array(obs["node_ent"][:len(obj_gnn_ind)])!=-1])
            obj_trans_ind += list(np.arange(num_ent)[:num_obj_ent] + prev_max_ent)
            gt_trans_ind += list(np.arange(num_ent)[num_obj_ent:] + prev_max_ent)
            prev_max_ents.append(prev_max_ent)
            prev_max_ent += num_ent
            # entity index -> problem index
            batch_ind += [ii] * sum(np.array(obs["node_ent"])!=-1)
            # print(sum(np.array(obs[2])!=-1))
        if cuda:
            batch_ind = torch.cuda.LongTensor(batch_ind)
        else:
            batch_ind = torch.LongTensor(batch_ind)

        batch_obj_state = dgl.batch(obj_gnns)
        if self.combined_gt_obj:
            obj_state_tensor, obj_out = self.gt_encoder(
                batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        else:
            obj_state_tensor, obj_out = self.obj_encoder(
                batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        if len(gt_gnns) > 0:
            batch_gt_state = dgl.batch(gt_gnns)
            gt_state_tensor, gt_out = self.gt_encoder(
                batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind)
            state_tensor = torch.cat((obj_state_tensor, gt_state_tensor), 0)
        else:
            gt_out = torch.zeros_like(obj_out)
            state_tensor = obj_state_tensor
        if gt_out.shape[0] < obj_out.shape[0]:
            zero_out_tensor = torch.zeros(obj_out.shape[0]-gt_out.shape[0], gt_out.shape[1]).to(device)
            gt_out = torch.cat((gt_out, zero_out_tensor), 0)
        out = torch.cat((obj_out, gt_out), 1)

        vf = self.vf_net(out)

        lemma_outputs = self.lemma_out(F.relu(self.lemma_q(out)))
        lemma_m = Categorical(probs=self.softmax(lemma_outputs))
        lemma_actions = lemma_m.sample()
        lemma_args = [theorem_no_input[index2thm[int(lemma)]] for lemma in lemma_actions]
        max_num_ent = 4
        masks = -1 * torch.ones(len(lemma_args), max_num_ent)
        for i in range(len(lemma_args)):
            for j in range(lemma_args[i]):
                masks[i][j] = 1
        masks = masks.to(device)

        batch_lemmas = one_hot(lemma_actions, len(thm2index))
        # TODO: add nonlinearities when merging info?
        out = F.relu(out.to(device) + F.relu(self.lemma_encoder(batch_lemmas).to(device)))

        if cuda:
            obj_node_ent = torch.cuda.LongTensor(obj_node_ent)
            gt_node_ent = torch.cuda.LongTensor(gt_node_ent)
            node_ent = torch.cuda.LongTensor(torch.cat((obj_node_ent, gt_node_ent), 0))
            ent_mask = (node_ent != -1).cuda()
            obj_trans_ind = torch.cuda.LongTensor(obj_trans_ind)
            gt_trans_ind = torch.cuda.LongTensor(gt_trans_ind)

            trans_ind = torch.cuda.LongTensor(torch.cat((obj_trans_ind, gt_trans_ind), 0))
            rev_trans_ind = torch.cuda.LongTensor(
                np.hstack([np.where(trans_ind.cpu().numpy() == i)[0] for i in range(len(trans_ind))]))
        else:
            obj_node_ent = torch.LongTensor(obj_node_ent)
            gt_node_ent = torch.LongTensor(gt_node_ent)
            node_ent = torch.LongTensor(torch.cat((obj_node_ent, gt_node_ent), 0))
            ent_mask = (node_ent != -1)
            obj_trans_ind = torch.LongTensor(obj_trans_ind)
            gt_trans_ind = torch.LongTensor(gt_trans_ind)

            trans_ind = torch.LongTensor(torch.cat((obj_trans_ind, gt_trans_ind), 0))
            rev_trans_ind = torch.LongTensor(
                np.hstack([np.where(trans_ind.cpu().numpy() == i)[0] for i in range(len(trans_ind))]))
        ent_rep = torch.index_select(state_tensor[ent_mask, :].clone(), 0, rev_trans_ind)


        # def sample(logits):
        #     noise = tf.random_uniform(tf.shape(logits))
        #     return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
        entity_actions = []
        h_uniform = Uniform(torch.zeros(ent_rep.shape[0]), torch.ones(ent_rep.shape[0]))
        # TODO: Check numerical issues from torch tensor to numpy
        for ii in range(max_num_ent):
            if cuda:
                mask = (masks[:, ii]!=-1).cuda()
            else:
                mask = (masks[:, ii]!=-1)
            if sum(mask) > 0:
                key = torch.index_select(out, 0, batch_ind)
                h = torch.mean(self.key_transform(key) * ent_rep, 1)
                gumble_h = h - torch.log(-torch.log(h_uniform.sample().to(device)))
                # calculate loss term.
                entity_action = scatter_max(gumble_h, batch_ind)[1]

                # use a zero tensor to fill in empty ent representation that
                # refers to an empty action slot
                zero_ind = torch.LongTensor(np.arange(entity_action.shape[0])).to(device)
                cur_ent_rep = torch.index_select(
                    ent_rep, 0, entity_action[mask].to(device))
                zero_tensor = torch.zeros(entity_action.shape[0], cur_ent_rep.shape[1]).to(device)
                cur_ent_rep = \
                    scatter_add(torch.cat((cur_ent_rep, zero_tensor), 0),
                                torch.cat((torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(device), zero_ind), 0), 0)
                # get new key vector
                out = F.relu(out + F.relu(self.ent_transform(cur_ent_rep)))
                # append entity actions; plus 1 because 0 is saved for no_op
                # entity_action = entity_action[mask] + 1 - torch.LongTensor(prev_max_ents)
                entity_action = (entity_action - torch.LongTensor(prev_max_ents).to(device))
                entity_action = ((entity_action.float() +1) * mask.float()).long()
                entity_action = torch.cat(
                    [entity_action,
                     torch.zeros(masks.shape[0]- entity_action.shape[0]).long().to(device)])
                entity_actions.append(entity_action)
        entity_actions = torch.stack(entity_actions).T
        # print([index2thm[int(lemma)] for lemma in lemma_actions])
        # print(lemma_args)

        entity_actions = torch.cat([entity_actions,
                                    torch.zeros([masks.shape[0],
                                                 masks.shape[1]- entity_actions.shape[1]]).long().to(device)], 1)
        actions = torch.cat([lemma_actions.view(-1, 1), entity_actions], 1)
        actions = actions.cpu().numpy()
        # print(actions)
        return actions, vf

    def batch_vf(self, observations):
        all_gnns = []
        obj_gnns = []
        obj_gnns_ind = []
        obj_batch_gnn_ind = []
        obj_prev_num_graphs = 0
        obj_node_ent = []
        obj_node_name = []
        gt_gnns = []
        gt_gnns_ind = []
        gt_batch_gnn_ind = []
        gt_prev_num_graphs = 0
        gt_node_ent = []
        gt_node_name = []
        batch_ind = []
        prev_max_ent = 0

        obj_trans_ind = []
        gt_trans_ind = []

        if isinstance(observations, np.ndarray) or isinstance(observations, list):
            observations = convert_batch_obs_to_dict(observations)

        for ii, obs in enumerate(observations):
            obj_gnn = obs["obj"]["obj_graph"]
            obj_gnn_ind = obs["obj"]["obj_gnn_ind"]
            # node index -> graph index
            obj_gnns_ind += [obj_prev_num_graphs+ind for ind in obj_gnn_ind]
            obj_prev_num_graphs = max(obj_gnns_ind) + 1
            obj_batch_gnn_ind += [ii] * (max(obj_gnn_ind) + 1)
            # graph index -> problem index
            obj_gnns += obj_gnn
            all_gnns += obj_gnn
            if len(obs["gt"]["gt_graph"]) > 0:
                gt_gnn = obs["gt"]["gt_graph"]
                gt_gnn_ind = obs["gt"]["gt_gnn_ind"]
                gt_gnns_ind += [gt_prev_num_graphs+ind for ind in gt_gnn_ind]
                gt_prev_num_graphs = max(gt_gnns_ind) + 1
                gt_batch_gnn_ind += [ii] * (max(gt_gnn_ind) + 1)
                all_gnns += gt_gnn
                gt_gnns += gt_gnn
            else:
                gt_gnn_ind = []

            assert len(obs["node_ent"]) == len(obj_gnn_ind) + len(gt_gnn_ind)
            obj_node_ent += obs["node_ent"][:len(obj_gnn_ind)]
            obj_node_name += obs["node_name"][:len(obj_gnn_ind)]
            gt_node_ent += obs["node_ent"][len(obj_gnn_ind):]
            gt_node_name += obs["node_name"][len(obj_gnn_ind):]

            num_ent = sum(np.array(obs["node_ent"])!=-1)
            num_obj_ent = len(np.array(obs["node_ent"][:len(obj_gnn_ind)])[np.array(obs["node_ent"][:len(obj_gnn_ind)])!=-1])
            obj_trans_ind += list(np.arange(num_ent)[:num_obj_ent] + prev_max_ent)
            gt_trans_ind += list(np.arange(num_ent)[num_obj_ent:] + prev_max_ent)
            # shift index for entity actions
            prev_max_ent += num_ent
            # entity index -> problem index
            batch_ind += [ii] * sum(np.array(obs["node_ent"])!=-1)
            # print(sum(np.array(obs[2])!=-1))

        batch_obj_state = Batch.from_data_list(obj_gnns)
        if self.combined_gt_obj:
            obj_state_tensor, obj_out = self.gt_encoder(
                batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        else:
            obj_state_tensor, obj_out = self.obj_encoder(
                batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        if len(gt_gnns) > 0:
            batch_gt_state = Batch.from_data_list(gt_gnns)
            gt_state_tensor, gt_out = self.gt_encoder(
                batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind)
        else:
            gt_out = torch.zeros_like(obj_out)
        if gt_out.shape[0] < obj_out.shape[0]:
            zero_out_tensor = torch.zeros(obj_out.shape[0]-gt_out.shape[0], gt_out.shape[1]).to(device)
            gt_out = torch.cat((gt_out, zero_out_tensor), 0)
        out = torch.cat((obj_out, gt_out), 1)

        vf = self.vf_net(out)
        return vf
