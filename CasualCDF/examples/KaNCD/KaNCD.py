# coding: utf-8

import logging
from EduCDM import KaNCD
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")
df_item = pd.read_csv("../../data/a0910/item.csv")

popularity_exp = -0.03
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

batch_size = 128
user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))


def transform(user, item, item2knowledge, score, diff, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32),
        torch.tensor(diff, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


train_set, valid_set, test_set = [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], data["diff"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)
cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)
#for i in range(21):
    #lr = 0.0001 + 0.0001 * i
cdm.train(train_set, test_set, epoch_n=20, device="cpu", lr=0.0034)
cdm.save("kancd.snapshot")

cdm.load("kancd.snapshot")
accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test_set, device="cpu")
print(' accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % ( accuracy, precision, rmse, recall, auc, f1))


