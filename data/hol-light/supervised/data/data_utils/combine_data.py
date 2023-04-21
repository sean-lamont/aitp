import pickle
import os
from tqdm import tqdm

files = os.listdir("train")
train_list = []

for file in tqdm(files):
    fpath = os.path.join('train', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    train_list.extend(x)

print (len(train_list))


with open("train_data.pk", "wb") as f:
    pickle.dump(train_list, f)


files = os.listdir("valid")
valid_list = []

for file in tqdm(files):
    fpath = os.path.join('valid', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    valid_list.extend(x)

print (len(valid_list))


with open("val_data.pk", "wb") as f:
    pickle.dump(valid_list, f)



files = os.listdir("test")
test_list = []

for file in tqdm(files):
    fpath = os.path.join('valid', file)

    with open(fpath, "rb") as f:
        x = pickle.load(f)

    test_list.extend(x)

print (len(test_list))


with open("data.pk", "wb") as f:
    pickle.dump(test_list, f)

