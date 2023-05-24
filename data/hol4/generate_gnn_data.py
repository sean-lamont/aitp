import re
from tqdm import tqdm
from data.hol4.ast_def import *
import data.hol4.ast_def as ast_def
from sklearn.preprocessing import OneHotEncoder

def gen_train_test_data(data, train_ratio=0.9, val_ratio=0.05, rand=True, deps=None, full_db=None):
    data_size = len(data)

    if rand:
        np.random.shuffle(data)

    train_exps = data[:int(train_ratio * data_size)]
    val_exps = data[int(train_ratio * data_size):int((train_ratio + val_ratio) * data_size)]
    test_exps = data[int((train_ratio + val_ratio) * data_size):]

    # generate positive samples
    es = []
    positive_data = []

    defs = [x for x in full_db.keys() if full_db[x][4] == "def"]

    #########################
    ### Training Data #######
    #########################

    for exp in train_exps:
        for dep in deps[exp]:
            try:
                # only consider dependencies in the training set
                # also include dependencies which are definitions, as these will all be included (in one hot encoding as well)
                if dep in train_exps or full_db[dep][4] == "def":
                    # exp is theorem, dep is useful to thm
                    positive_data.append((full_db[exp][2], full_db[dep][2], 1))

            # exception for dependencies without database entry, caused from full expression not appearing in HOL deps
            except Exception as e:
                es.append(e)

    # generate negative samples

    # this procedure is far from ideal - negative samples may indeed be useful for proving the paired theorem but simply not
    # used in the recorded dependencies. On average, most theorems should not be useful however (remains to be seen experimentally)

    # valid keys are only expressions from training set again, since training procedure hides nodes from val/test set

    neg_data = []

    candidate_deps = train_exps + defs
    # train_shuff = copy.deepcopy(train_exps)

    # generate one hot encoder for training data

    enc_nodes = OneHotEncoder(handle_unknown='ignore')

    enc_nodes.fit(np.array([full_db[j][2] for j in candidate_deps]).reshape(-1, 1))

    e = enc_nodes.transform(np.array([full_db[j][2] for j in candidate_deps]).reshape(-1, 1))

    preds = enc_nodes.inverse_transform(e)

    # ensure encoding is correct
    assert [preds[:, 0][i] for i in range(preds.shape[0])] == [full_db[j][2] for j in candidate_deps]

    # generate single negative example for each positive
    for i in tqdm(range(len(positive_data))):

        rand_key = train_exps[np.random.randint(0, len(train_exps))]

        # should allowed theories be restricted to those used in the expression?
        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', full_db[rand_key][2])))

        if "min" in allowed_theories:
            allowed_theories.remove("min")

        # generate valid keys to sample from

        valid_keys = []

        np.random.shuffle(candidate_deps)

        valid_key = None
        for t in candidate_deps:
            if full_db[t][0] in allowed_theories and (
                    full_db[t][0] != full_db[rand_key][0] or int(full_db[t][3]) < int(full_db[rand_key][3])):
                try:
                    # ensure not dependency so it's a negative sample
                    if t not in deps[rand_key]:
                        valid_key = t
                        break
                # hack while all theorems aren't in db
                except:
                    continue

        if valid_key is None:
            continue

        neg_data.append((rand_key, valid_key))

    neg_samples = [(full_db[x[0]][2], full_db[x[1]][2], 0) for x in neg_data]

    train_data = positive_data + neg_samples

    # np.random.shuffle(whole_data)

    # repeat for val and test sets

    #########################
    ### Validation Data #####
    #########################

    positive_data = []

    for exp in val_exps:

        for dep in deps[exp]:
            try:
                if dep in train_exps or full_db[dep][4] == "def":
                    positive_data.append((full_db[exp][2], full_db[dep][2], 1))
            except Exception as e:
                es.append(e)

    neg_data = []

    for i in tqdm(range(len(positive_data))):

        rand_key = val_exps[np.random.randint(0, len(val_exps))]

        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', full_db[rand_key][2])))

        if "min" in allowed_theories:
            allowed_theories.remove("min")

        # generate valid keys to sample from

        valid_keys = []

        np.random.shuffle(candidate_deps)

        valid_key = None
        for t in candidate_deps:
            if full_db[t][0] in allowed_theories and (
                    full_db[t][0] != full_db[rand_key][0] or int(full_db[t][3]) < int(full_db[rand_key][3])):
                try:
                    # ensure not dependency so it's a negative sample
                    if t not in deps[rand_key]:
                        valid_key = t
                        break
                # hack while all theorems aren't in db
                except:
                    continue

        if valid_key is None:
            continue

        neg_data.append((rand_key, valid_key))

    neg_samples = [(full_db[x[0]][2], full_db[x[1]][2], 0) for x in neg_data]

    val_data = positive_data + neg_samples
    #########################
    ### Test Data ###########
    #########################
    positive_data = []
    for exp in test_exps:

        for dep in deps[exp]:
            try:
                if dep in train_exps or full_db[dep][4] == "def":
                    positive_data.append((full_db[exp][2], full_db[dep][2], 1))
            except Exception as e:
                es.append(e)

    neg_data = []

    for i in tqdm(range(len(positive_data))):

        rand_key = test_exps[np.random.randint(0, len(test_exps))]

        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', full_db[rand_key][2])))

        if "min" in allowed_theories:
            allowed_theories.remove("min")

        # generate valid keys to sample from

        valid_keys = []

        np.random.shuffle(candidate_deps)

        valid_key = None
        for t in candidate_deps:
            if full_db[t][0] in allowed_theories and (
                    full_db[t][0] != full_db[rand_key][0] or int(full_db[t][3]) < int(full_db[rand_key][3])):
                try:
                    # ensure not dependency so it's a negative sample
                    if t not in deps[rand_key]:
                        valid_key = t
                        break
                # hack while all theorems aren't in db
                except:
                    continue

        if valid_key is None:
            continue

        neg_data.append((rand_key, valid_key))

    neg_samples = [(full_db[x[0]][2], full_db[x[1]][2], 0) for x in neg_data]

    test_data = positive_data + neg_samples

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data, enc_nodes

def generate_gnn_data(data, train_ratio, val_ratio, rand, data_dir,deps,full_db):
    print ("Generating train, val, test (goal, premise, label) pairs...")
    whole_data = gen_train_test_data(data, train_ratio, val_ratio, rand, deps, full_db)
    with open(data_dir+"train_test_data.pk", 'wb') as f:
        pickle.dump(whole_data, f)

    # get all unique tokens in the full database
    tokens = list(
        set([token.value for polished_goal in full_db.keys() for token in polished_to_tokens_2(full_db[polished_goal][2]) if
             token.value[0] != 'V']))

    # add tokens once variables and variable functions are abstracted, and for unseen tokens
    tokens.append("VAR")
    tokens.append("VARFUNC")
    tokens.append("UNKNOWN")

    #todo map unknown to "UNKNOWN" token
    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(np.array(tokens).reshape(-1, 1))

    e = enc.transform(np.array(tokens).reshape(-1, 1))

    preds = enc.inverse_transform(e)
    # ensure encoding is correct
    assert [preds[:, 0][i] for i in range(preds.shape[0])] == tokens

    polished_goals = [full_db[k][2] for k in full_db.keys()]
    torch_graph_dict = {}

    print ("Converting goals to graphs..")
    for goal in tqdm(polished_goals):
        torch_graph_dict[goal] = ast_def.graph_to_torch_labelled(goal_to_graph_labelled(goal), enc)

    one_hot_dict = {}

    # enc_nodes = whole_data[3]

    # print ("Generating one hot labels for expressions in HOL4 graph...")
    # for key in tqdm(torch_graph_dict):
    #     one_hot_dict[key] = enc_nodes.transform([[key]])

    # with open(data_dir+"one_hot_dict.pk", "wb") as f:
    #     pickle.dump(one_hot_dict, f)

    with open(data_dir+"torch_graph_dict.pk", "wb") as f:
        pickle.dump(torch_graph_dict, f)

    with open(data_dir+"graph_token_encoder.pk", "wb") as f:
        pickle.dump(enc, f)

    # todo add directed edge info + depth info to torch_graph_dict




# if __name__ == "__main__":
#     print ("Generating GNN premise selection dataset using all theorems from database...")
#
#
#     unique_thms = list(set(deps.keys()))
#     generate_gnn_data(unique_thms, 0.9, 0.05, True, "")



# graph-of-graphs
# edge cases, multiple equivalent polished expressions map to different theory name-numbers e.g. list-8 <-> bool-25
# can be confirmed from processed_old HOL file include_probability.txt

# # map from polished goal to library-dep key
# with open("polished_dict.json") as f:
#     p_d = json.load(f)
# new_pd = {}

# for key, val in p_d.items():

#     if val[0] == " ":
#         #new_db.pop(key)
#         new_pd[key] = val[1:]
#     else:
#         new_pd[key] = val

# count = 0
# for (a, b, y) in train:
#     try:
#         if new_pd[b] in deps[new_pd[a]]:
#             if y == 0:
#                 print (y)
#                 print (new_pd[b])
#                 print (deps[new_pd[a]])
#     except:
#         continue
#     try:
#         if new_pd[b] not in deps[new_pd[a]]:
#             if y == 1:
#                 count += 1
#                 print (y)
#                 print (a)
#                 print (new_pd[b])
#                 print (new_pd[a])
#                 print (deps[new_pd[a]])
#     except:
#         continue
