import pickle
import os
from tqdm import tqdm

data_dir = "../graph_data/"


files = os.listdir(data_dir + "train")
train_list = []

for file in tqdm(files):
    fpath = os.path.join(data_dir,'train', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    train_list.extend(x)

print (f"Train samples: {len(train_list)}")


with open(data_dir + "train_data.pk", "wb") as f:
    pickle.dump(train_list, f)


files = os.listdir(data_dir + "valid")
valid_list = []

for file in tqdm(files):
    fpath = os.path.join(data_dir, 'valid', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    valid_list.extend(x)

print (f"Valid samples: {len(valid_list)}")


with open(data_dir + "val_data.pk", "wb") as f:
    pickle.dump(valid_list, f)



files = os.listdir(data_dir + "test")
test_list = []

for file in tqdm(files):
    fpath = os.path.join(data_dir,'test', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    test_list.extend(x)


print (f"Test samples: {len(test_list)}")

with open(data_dir + "test_data.pk", "wb") as f:
    pickle.dump(test_list, f)

