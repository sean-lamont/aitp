import numpy as np
from tqdm import tqdm
import json
from pymongo import MongoClient
import pickle
import torch

data_dir = "data/"

with open(data_dir + "dep_data.json") as f:
    dep_data = json.load(f)

with open(data_dir + "paper_goals.pk", "rb") as f:
    paper_dataset = pickle.load(f)

with open(data_dir + "new_db.json") as f:
    full_db = json.load(f)

with open(data_dir + "torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

with open(data_dir + "train_test_data.pk", "rb") as f:
    train_test_data = pickle.load(f)

with open(data_dir + "graph_token_encoder.pk", "rb") as f:
    token_enc = pickle.load(f)

new_db = {v[2]: v for k, v in full_db.items()}

# Issue with multiple expressions with the same polished value in paper goals.
# Temporary fix to just take the plain paper goal term whenever it appears in new_db
for goal in paper_dataset:
    if goal[0] in new_db:
        new_db[goal[0]][5] = goal[1]

with open(data_dir + "adjusted_db.json", "w") as f:
    json.dump(new_db, f)

# Find goals from original paper which have a database entry
valid_goals = []
for goal in paper_dataset:
    if goal[0] in new_db.keys():
        valid_goals.append(goal)

print(f"Len valid {len(valid_goals)}")
np.random.shuffle(valid_goals)

with open(data_dir + "valid_goals_shuffled.pk", "wb") as f:
    pickle.dump(valid_goals, f)

# Database used for replicating TacticZero, and for pretraining using HOL4 dependency information.
# Contains information from the HOL4 standard library up to and including "probabilityTheory"
db_name = "hol4_tactic_zero"

# Collection containing meta information about an expression (library, theorem name, etc.)
info_name = "expression_metadata"

# Collection containing dependency information for expressions
dep_name = "dependency_data"

# Collection containing (goal, premise) pairs for pretraining
pretrain_name = "pretrain_data"

# Collection containing the goals from original paper, shuffled
paper_name = "paper_goals"

# Collection mapping polished expression to graph representation (one-hot indices, edge index, edge attributes)
expression_graph_name = "expression_graph_data"

# Collection mapping token to one-hot index
vocab_name = "vocab"

db_client = MongoClient()

db = db_client[db_name]

dependency_data = db[dep_name]
pretrain_data = db[pretrain_name]
paper_split = db[paper_name]
expression_graph_data = db[expression_graph_name]
vocab = db[vocab_name]
expression_info_data = db[info_name]

print(f"Adding HOL4 standard library data up to and including \"probabilityTheory\" to database {db_name}\n")

for k, v in tqdm(dep_data.items()):
    info = dependency_data.insert_one(
        {"_id": k,
         "dependencies": v})

for (k, v) in tqdm(torch_graph_dict.items()):
    test_graph = v
    tok_inds = list(test_graph.x.coalesce().indices().to_dense()[1].tolist())
    # print(tok_inds, test_graph.edge_index.tolist(), test_graph.edge_attr.long().tolist(), test_graph.labels)
    info = expression_graph_data.insert_one(
        {"_id": k,
         "graph": {"onehot": tok_inds, "edge_index": test_graph.edge_index.tolist(),
                   "edge_attr": test_graph.edge_attr.long().tolist(), "labels": test_graph.labels}})

train, val, test, enc_nodes = train_test_data

for conj, stmt, y in tqdm(train):
    info = pretrain_data.insert_one(
        {"split": "train", "conj": conj, "stmt": stmt, "y": y})

for conj, stmt, y in tqdm(val):
    info = pretrain_data.insert_one(
        {"split": "val", "conj": conj, "stmt": stmt, "y": y})

for conj, stmt, y in tqdm(test):
    info = pretrain_data.insert_one(
        {"split": "test", "conj": conj, "stmt": stmt, "y": y})

mapping = {i: v for i, v in enumerate(token_enc.categories_[0])}

for k, v in tqdm(mapping.items()):
    info = vocab.insert_one({"_id": k, "token": v})

for k, v in tqdm(new_db.items()):
    info = expression_info_data.insert_one(
        {"_id": k, "theory": v[0], "name": v[1], "dep_id": v[3], "type": v[4], "plain_expression": v[5]})

info = paper_split.insert_many([{"_id": g[0], "plain": g[1]} for g in valid_goals])
