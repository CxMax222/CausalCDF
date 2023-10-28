# coding: utf-8
# 2021/3/23 @ tongshiwei

import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score



class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        return torch.squeeze(torch.sigmoid(torch.sum(torch.mul(user, item), dim=-1)))
            #torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)



class MCD(CDM):
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MCD, self).__init__()
        self.mf_net = MFNet(user_num, item_num, latent_dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.mf_net = self.mf_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.mf_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.mf_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                accuracy, precision, rmse, recall, auc, f1 = self.eval(test_data, device=device)
                print('epoch= %d, accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    e, accuracy, precision, rmse, recall, auc, f1))
    def eval(self, test_data, device="cpu") -> tuple:
        self.mf_net = self.mf_net.to(device)
        self.mf_net.eval()
        y_pred = []
        y_true = []
        pred_class_all = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.mf_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

            batch_pred_class = []
            batch_ys = response
            batch_pred = pred
            for i in range(len(batch_ys)):
                if batch_pred[i] >= 0.5:
                    batch_pred_class.append(1)
                else:
                    batch_pred_class.append(0)
            pred_class_all += batch_pred_class

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        pred_class_all = np.array(pred_class_all)
        accuracy = accuracy_score(y_true, pred_class_all)
        precision = precision_score(y_true, pred_class_all)
        recall = recall_score(y_true, pred_class_all)
        f1 = f1_score(y_true, pred_class_all)

        # compute RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        # compute AUC
        auc = roc_auc_score(y_true, y_pred)

        self.mf_net.train()
        return accuracy, precision, rmse, recall, auc, f1


    def save(self, filepath):
        torch.save(self.mf_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.mf_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
