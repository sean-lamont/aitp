import pickle

data_dir = "../graph_data/"

with open(data_dir + "test/expr_dict", "rb") as f:
    test_dict = pickle.load(f)

with open(data_dir + "train/expr_dict", "rb") as f:
    train_dict = pickle.load(f)

with open(data_dir + "valid/expr_dict", "rb") as f:
    val_dict = pickle.load(f)

print (len(test_dict))
print (len(train_dict))
print (len(val_dict))


train_dict.update(test_dict)

print (len(train_dict))

train_dict.update(val_dict)

print (len(train_dict))


with open(data_dir + "global_expr_dict.pk", "wb") as f:
    pickle.dump(train_dict, f)
