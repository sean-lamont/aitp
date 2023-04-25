import numpy as np
import json
from pymongo import MongoClient
import pickle
import torch
from environments.hol4.new_env import *
db_client = MongoClient()

with open("dep_data.json") as f:
    dep_data = json.load(f)

with open("paper_goals.pk", "rb") as f:
    paper_dataset = pickle.load(f)

with open("new_db.json") as f:
    full_db = json.load(f)

with open("torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

with open("train_test_data.pk", "rb") as f:
    train_test_data = pickle.load(f)

with open("graph_token_encoder.pk", "rb") as f:
    token_enc = pickle.load(f)


print ("dep:")
for k,v in dep_data.items():
    print (k,v)
    break

print ("paper ")
for k in paper_dataset:
    print (k)
    break


# todo merge with old "include_probability.json"
# todo determine best approach when dependencies are inconsistent between the two...?
print ("full_db")
for (k,v) in full_db.items():
    print (k,v)
    break


print ("torch graph")

for (k,v) in torch_graph_dict.items():
    print (k,v)
    test_graph = v
    tok_inds = list(test_graph.x.coalesce().indices().to_dense()[1].numpy())
    print (tok_inds, test_graph.edge_index.tolist(), test_graph.edge_attr.long().tolist(), test_graph.labels)
    break

print ("train_test")
train, val, test, enc_nodes = train_test_data



print ("one-hot enc")

mapping = {i : v for i,v in enumerate(token_enc.categories_[0])}

print ([mapping[tok] for tok in tok_inds])

#todo test out RL loop with new database
#todo add these all to separate collections in a HOL4 MongoDB e.g. "standard library", as this is all expressions from the standard library up to prob theory
#todo test RL loop with MongoDB
#todo using other HOL4 sources e.g. CakeML can be done later, with integration into a 'full' database of everything seen in HOL4



new_db = {v[2]: v for k,v in full_db.items()}
# issue with multiple expressions with the same polished value in paper goals. Temporary fix to just take the plain paper goal term whenever it appears in new_db
for goal in paper_dataset:
    if goal[0] in new_db:
        new_db[goal[0]][5] = goal[1]
with open("adjusted_db.json", "w") as f:
    json.dump(new_db, f)
valid_goals = []
for goal in paper_dataset:
    if goal[0] in new_db.keys():
        valid_goals.append(goal)

print (f"Len valid {len(valid_goals)}")
np.random.shuffle(valid_goals)

with open("valid_goals_shuffled.pk", "wb") as f:
    pickle.dump(valid_goals, f)


# with open("include_probability.json") as f:
#     old_db = json.load(f)
# new_keys = [full_db[k][2] for k in full_db]
# new_keys = list(new_db.keys())
# cnt = Counter(new_keys)

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
# env = HolEnv("T")
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
#         print (f"Error in new {goal_new}")
#         new_errs.append(goal_new)
#     try:
#         p_goal = env.get_polish(goal_old)
#     except:
#         print (f"Error in old {goal_old}")
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
#         except Exception as e:
#             print (e)
#             paper_errs_new_db.append((paper_goal))
#
#     if paper_goal in old_keys:
#         try:
#             env.get_polish(old_db[paper_goal][4])
#         except Exception as e:
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