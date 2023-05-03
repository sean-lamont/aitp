from environments.hol4.new_env import *
import json
import pickle
from data.hol4 import generate_gnn_data
import os


data_dir = "data/hol4/data/"
# data_dir = "data/"
data_dir = os.path.join(os.getcwd(),data_dir)
paper_dir = os.path.join(data_dir, "paper_goals.pk")

if os.path.exists(paper_dir):
    with open(paper_dir, "rb") as f:
        paper_goals = pickle.load(f)
else:
    print("Generating original goals from TacticZero paper...")

    with open(data_dir + "dataset.json") as fp:
        dataset = json.load(fp)

    env = HolEnv("T")
    paper_goals = []

    for goal in dataset:
        try:
            p_goal = env.get_polish(goal)
            paper_goals.append((p_goal[0]["polished"]['goal'], goal))
        except:
            print (f"Unable to process Goal {goal}")

    print (f"Processed {len(paper_goals)}/{len(dataset)} paper goals")

    with open(data_dir + "paper_goals.pk", "wb") as f:
        pickle.dump(paper_goals, f)

with open(data_dir + "dep_data.json") as fp:
    deps = json.load(fp)

with open(data_dir + "new_db.json") as fp:
    full_db = json.load(fp)

with open(data_dir + "include_probability.json") as fp:
    old_db = json.load(fp)

unique_thms = list(set(deps.keys()))

paper_goals_polished = [g[0] for g in paper_goals]

# get all theorems from paper dataset compatible with current database
# todo link with compat_db used in RL

exp_thms = []

for thm in unique_thms:
    if full_db[thm][2] in paper_goals_polished:
        exp_thms.append(thm)

#remove theorems the RL agent trains/tests on from those used to pretrain the GNN encoder
gnn_encoder_set = list(set(unique_thms) - set(exp_thms))

# print (len(set(exp_thms)), len(set(unique_thms)), len(gnn_encoder_set))

# train_thms = exp_thms[:int(0.8 * len(exp_thms))]
# test_thms = exp_thms[int(0.8 * len(exp_thms)):]

#generate gnn data from valid set excluding goals for RL agent

print ("Generating graph dataset for pretraining...")

generate_gnn_data.generate_gnn_data(gnn_encoder_set, 0.95, 0.05, True, data_dir, deps, full_db)