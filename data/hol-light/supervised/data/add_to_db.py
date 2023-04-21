import torch
from pymongo import MongoClient
from tqdm import tqdm

with open("hol_train_dict", "rb") as f:
	hol_dict = torch.load(f)

client = MongoClient()
db = client.aitp
collection = db.hol_light

for k,v in tqdm(hol_dict.items()):
	info = collection.insert_one({"type":"supervised", "task":"holstep", "name":"hol_train_dict", "token":k, "value":v})


