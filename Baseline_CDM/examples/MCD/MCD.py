# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from EduCDM import MCD
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#rd.seed(seed)

train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")

batch_size = 256


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = MCD(4164, 17747, 128)

cdm.train(train, test, epoch=10)
cdm.save("mcd.params")

cdm.load("mcd.params")
accuracy, precision, rmse, recall, auc, f1 = cdm.eval(test)
print('accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    accuracy, precision, rmse, recall, auc, f1))

