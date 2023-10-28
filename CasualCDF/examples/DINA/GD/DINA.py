# coding: utf-8

import logging
from EduCDM import GDDINA
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#rd.seed(seed)



train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")
item_data = pd.read_csv("../../../data/a0910/item.csv")

popularity_exp = -3
def diff_power(k):
    if k == 0:
        k = 1e-6
    p = np.power(k, popularity_exp)
    return p

train_data["diff"] = train_data["diff"].apply(diff_power)
valid_data["diff"] = valid_data["diff"].apply(diff_power)
test_data["diff"] = test_data["diff"].apply(diff_power)

knowledge_num = 123

def code2vector(x):
    vector = [0] * knowledge_num
    for k in eval(x):
        vector[k - 1] = 1
    return vector

item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)

train_data = pd.merge(train_data, item_data, on="item_id")
valid_data = pd.merge(valid_data, item_data, on="item_id")
test_data = pd.merge(test_data, item_data, on="item_id")
#print(train_data)

batch_size = 256


def transform(x, y, z, k, diff, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(diff, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], data["knowledge"], data["diff"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = GDDINA(4164, 17747, knowledge_num)

cdm.train(train, test, epoch=8, lr=0.002)
cdm.save("dina.params")

cdm.load("dina.params")

accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test)
print('accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    accuracy, precision, rmse, recall, auc, f1))