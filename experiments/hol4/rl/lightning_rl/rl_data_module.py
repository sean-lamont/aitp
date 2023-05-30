from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as loader
import lightning.pytorch as pl
import torch.optim
from data.hol4.ast_def import graph_to_torch_labelled
from data.hol4 import ast_def
from environments.hol4.new_env import *

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

def gather_encoded_content_gnn(history, encoder, device, graph_db, token_enc, data_type='graph'):
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

    if data_type == 'graph':
        loader = DataLoader(graphs, batch_size=len(graphs))
        batch = next(iter(loader))
        batch.to(device)
        batch.edge_attr = batch.edge_attr.long()

        # for batch norm, when only one goal
        encoder.eval()
        representations = torch.unsqueeze(encoder(batch), 1)
        encoder.train()

    elif data_type == 'relation':
        graphs = data_to_relation(graphs)
        graphs.to(device)
        representations = torch.unsqueeze(encoder(graphs), 1)

    return representations, contexts, fringe_sizes

class RLData(pl.LightningDataModule):
    def __init__(self, train_goals, test_goals, config, database=None, graph_db=None):
        super().__init__()
        self.config = config
        self.env = HolEnv("T")
        self.train_goals = train_goals
        self.test_goals = test_goals
        self.database = database
        self.graph_db = graph_db

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
            # print("Removed simpset of {}".format(goal_theory))

        except Exception as e:
            raise Exception(f"Error generating fact pool: {e}")

        graphs = [self.graph_db[t] for t in candidate_args]

        if self.config['data_type'] == 'graph':
            loader = DataLoader(graphs, batch_size=len(candidate_args))
            allowed_fact_batch = next(iter(loader))
            allowed_fact_batch.edge_attr = allowed_fact_batch.edge_attr.long()

        elif self.config['data_type'] == 'relation':
            allowed_fact_batch = data_to_relation(graphs)

        #todo not sure if we need allowed_arguments_ids?
        return allowed_fact_batch, allowed_arguments_ids, candidate_args

    def setup_goal(self, goal):
        goal = goal[0]
        try:
            self.env.reset(goal[1])
        except:
            self.env = HolEnv("T")
            return None
        allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(self.env, goal)
        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, self.env

    def train_dataloader(self):
        return loader(self.train_goals, batch_size=1, collate_fn=self.setup_goal)

    # todo: val set to terminate training with??
    def test_dataloader(self):
        return loader(self.test_goals, collate_fn=self.setup_goal)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None

        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        allowed_fact_batch = allowed_fact_batch.to(device)

        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env



# test model
# module = RLData(train_goals=train_goals, test_goals=test_goals, database=compat_db, graph_db=graph_db)
# module.setup("fit")
# batch = next(iter(module.train_dataloader()))
# print (batch)
