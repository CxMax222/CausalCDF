# coding: utf-8
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from data_loader import transform
from model import NCDM

train_data = pd.read_csv("../data/train.csv")
valid_data = pd.read_csv("../data/valid.csv")
test_data = pd.read_csv("../data/test.csv")
df_item = pd.read_csv("../data/item.csv")

popularity_exp = -2
def diff_power(k):
    if k == 0:
        k = 1e-10
    p = np.power(k, popularity_exp)
    return p

train_data["diff"] = train_data["diff"].apply(diff_power)
valid_data["diff"] = valid_data["diff"].apply(diff_power)
test_data["diff"] = test_data["diff"].apply(diff_power)

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 32
user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))
print(user_n,item_n)


train_set, valid_set, test_set = [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], data["diff"], batch_size,knowledge_n)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)
cdm = NCDM(knowledge_n, item_n, user_n)
cdm.train(train_set, test_set, epoch=5, device="cpu", lr=0.001)
cdm.save("ncdm.snapshot")

cdm.load("ncdm.snapshot")
accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test_set)
print(' accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % ( accuracy, precision, rmse, recall, auc, f1))

