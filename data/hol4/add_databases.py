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


#todo test RL loop with MongoDB
#todo using other HOL4 sources e.g. CakeML can be done later, with integration into a 'full' database of everything seen in HOL4

new_db = {v[2]: v for k,v in full_db.items()}

# issue with multiple expressions with the same polished value in paper goals. Temporary fix to just take the plain paper goal term whenever it appears in new_db
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

print (f"Len valid {len(valid_goals)}")
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

print (f"Adding HOL4 standard library data up to and including \"probabilityTheory\" to database {db_name}\n")


# print (f"Adding dependency data to collection {dep_name}\n")

for k,v in tqdm(dep_data.items()):
    info = dependency_data.insert_one(
        {"_id": k,
         "dependencies": v})

# print (f"Adding expression graph data to collection {expression_graph_name}\n")


for (k, v) in tqdm(torch_graph_dict.items()):
    test_graph = v
    tok_inds = list(test_graph.x.coalesce().indices().to_dense()[1].tolist())
    # print(tok_inds, test_graph.edge_index.tolist(), test_graph.edge_attr.long().tolist(), test_graph.labels)
    info = expression_graph_data.insert_one(
        {"_id": k,
         "graph": {"onehot": tok_inds, "edge_index":test_graph.edge_index.tolist(),"edge_attr":test_graph.edge_attr.long().tolist(), "labels": test_graph.labels}})

# print (f"Adding premise selection split data to collection {pretrain_name}\n")

train, val, test, enc_nodes = train_test_data

for conj, stmt, y in tqdm(train):
    info = pretrain_data.insert_one(
        {"split": "train", "conj":conj, "stmt":stmt, "y":y})

for conj, stmt, y in tqdm(val):
    info = pretrain_data.insert_one(
        {"split": "val", "conj":conj, "stmt":stmt, "y":y})

for conj, stmt, y in tqdm(test):
    info = pretrain_data.insert_one(
        {"split": "test", "conj":conj, "stmt":stmt, "y":y})


# print (f"Adding vocab data to collection {vocab_name}\n")

mapping = {i : v for i,v in enumerate(token_enc.categories_[0])}

for k,v in tqdm(mapping.items()):
    info = vocab.insert_one({"_id": k, "token":v})

# print (f"Adding expression metadata to collection {info_name}\n")

for k,v in tqdm(new_db.items()):
    info = expression_info_data.insert_one({"_id": k, "theory":v[0],"name":v[1], "dep_id":v[3], "type":v[4], "plain_expression":v[5]})

# print (f"Adding valid paper goals to collection {paper_name}\n")

info = paper_split.insert_many([{"_id": g[0], "plain":g[1]} for g in valid_goals])

# with open("include_probability.json") as f:
#     old_db = json.load(f)
# new_keys = [full_db[k][2] for k in full_db]
# new_keys = list(new_db.keys())
# cnt = counter(new_keys)

# print ([k for k, v in cnt.items() if v > 1])
# num_dups = []
# paper_polished = [goal[0] for goal in paper_dataset]

# for k,v in cnt.items():
#     if v > 1 and k in paper_polished:
#         paper_plain = paper_dataset[paper_polished.index(k)][1]
#         new_db[k][5] = paper_plain

        # for k1,v1 in full_db.items():
        #     if v1[2] == k:
        #         if v1[5] != paper_dataset[paper_polished.index(k)][1]:
        #             print (v1[5])
        #             print (paper_dataset[paper_polished.index(k)][1])
        #             print (old_db[k][4])
        #             print (v1[5] == old_db[k][4])
                    # num_dups.append(v1)
                # print (v1)

# todo what about dependencies?
#%%
# missing = 0
# for goal in paper_dataset:
#     if goal[0] in new_db:
#         if new_db[goal[0]][5] != goal[1]:
#             print (goal, new_db[goal[0]])
#     else:
#         missing += 1

#%%
# missing
#%%
# env = holenv("t")
#%%
# goals with different polished goal but same theorem / definition name
# goals_to_test = []
# for key in old_keys:
#     if key not in new_keys:
#         for k1,v1 in full_db.items():
#             if v1[1] == old_db[key][1]:
#                 goals_to_test.append((v1[5],old_db[key][4]))
#
# others = []
# for key, value in full_db.items():
#     if value[2] not in old_keys:
#         for k1,v1 in old_db.items():
#             if v1[1] == value[1]:
#                 others.append((value[5],v1[4]))

#%%
# len(goals_to_test)
# len(goals_to_test)
#%%
from tqdm import tqdm
#%%
# old_errs = []
# new_errs = []
# for goal_new, goal_old in tqdm(goals_to_test + others):
#     try:
#         p_goal = env.get_polish(goal_new)
#     except:
#         print (f"error in new {goal_new}")

#         new_errs.append(goal_new)
#     try:
#         p_goal = env.get_polish(goal_old)
#     except:
#         print (f"error in old {goal_old}")
#         old_errs.append(goal_old)
#%%
# many more errors in expressions from the old database
# len(new_errs) # = 32
# len(old_errs) # = 120
#%%


#check which paper_goals are best between databases
# paper_errs_new_db = []
# paper_errs_old_db = []
#
# for goal in paper_dataset:
#     paper_goal = goal[0]
#
#     if paper_goal in new_keys:
#         try:
#             env.get_polish(old_rev[paper_goal][5])
#         except exception as e:
#             print (e)
#             paper_errs_new_db.append((paper_goal))
#
#     if paper_goal in old_keys:
#         try:
#             env.get_polish(old_db[paper_goal][4])
#         except exception as e:
#             print (e)
#             paper_errs_old_db.append((paper_goal))
#

#%%
# len(paper_errs_old_db)
#%%
# issue different plain expressions for the same polished term between paper goals and both databases. can possibly just replace database entry with plain goal from paper dataset
# inconsistent_old = []
# inconsistent_new = []
#
# for goal in paper_dataset:
#
#     if goal[0] in old_db:
#         old_plain = old_db[goal[0]][4]
#         if old_plain != goal[1]:
#             inconsistent_old.append((goal[0], old_plain, goal[1]))
#             # print (goal[0])
#             # print ('old')
#             # print (old_plain)
#             # print (goal[1])
#
#
#     if goal[0] in old_rev:
#         new_plain = old_rev[goal[0]][5]
#         if new_plain != goal[1]:
#             inconsistent_new.append((goal[0], new_plain, goal[1]))
            # print (goal[0])
            # print ('new')
            # print (new_plain)
            # print (goal[1])


# print (old_db[paper_dataset[0][0]][4])
# print (paper_dataset[0][1])