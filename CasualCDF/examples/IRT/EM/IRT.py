# coding: utf-8

import logging
from EduCDM import NCDM_IRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random as rd

#rd.seed(2022)
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rd.seed(seed)

train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")
df_item = pd.read_csv("../../../data/a0910/item.csv")
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

def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    data_loader = New_Dataloader(data_set,batch_size)
    data_loader.reset()
    final_data = []

    while not data_loader.is_end():
        final_data.append(data_loader.next_batch())


    return final_data


class New_Dataloader(object):
    def __init__(self,data_set,batch_size):
        self.data_set = data_set
        self.batch_size = batch_size
        self.ptr = 0


    def next_batch(self):
        if self.is_end():
            return None
        batch = [rd.choice(self.data_set) for _ in range(self.batch_size)]

        self.ptr += self.batch_size
        return batch

    def is_end(self):
        if self.ptr >= len(self.data_set):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0



train_set, valid_set, test_set = [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]


logging.getLogger().setLevel(logging.INFO)
cdm = NCDM_IRT.NCDM_IRT(knowledge_n, item_n, user_n)
cdm.train(train_set, valid_set, epoch=10, device="cpu")
cdm.save("ncdm.snapshot")

cdm.load("ncdm.snapshot")
accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test_set)
print('accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    accuracy, precision, rmse, recall, auc, f1))


