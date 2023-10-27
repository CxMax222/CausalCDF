# coding: utf-8
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np




def transform(user, item, item2knowledge, score, diff, batch_size,knowledge_n):
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



