# coding: utf-8

import logging
from EduCDM import MIRT
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

train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")

popularity_exp = -2
def diff_power(k):
    if k == 0:
        k = 1e-10
    p = np.power(k, -popularity_exp)
    return p

train_data["diff"] = train_data["diff"].apply(diff_power)
valid_data["diff"] = valid_data["diff"].apply(diff_power)
test_data["diff"] = test_data["diff"].apply(diff_power)
#print(train_data)


batch_size = 256



def transform(x, y, z, diff, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(diff, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], data['diff'], batch_size)
    for data in [train_data, valid_data, test_data]
]


logging.getLogger().setLevel(logging.INFO)

cdm = MIRT(4164, 17747, 123)

cdm.train(train, test, epoch=10, lr=0.001)
cdm.save("mirt.params")

cdm.load("mirt.params")
accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test)
print('accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    accuracy, precision, rmse, recall, auc, f1))

