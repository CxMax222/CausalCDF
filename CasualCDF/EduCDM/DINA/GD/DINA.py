# coding: utf-8

import csv
import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)
        self.out = nn.Linear(1, 1)

        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user, item, knowledge, diff, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            scores = torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            scores = (1 - slip) ** n * guess ** (1 - n)
        #print(scores.shape)
        #_scores = nn.ELU()(scores) + 1
        scores_with_diff = torch.mul(scores.unsqueeze(1), diff.unsqueeze(1))
        out = self.out(scores_with_diff)
        #print(self.out.state_dict())
        out = torch.squeeze(torch.sigmoid(out), dim=-1)
        return out


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DINA(CDM):
    def __init__(self, user_num, item_num, hidden_dim, ste=False):
        super(DINA, self).__init__()
        if ste:
            self.dina_net = STEDINANet(user_num, item_num, hidden_dim)
        else:
            self.dina_net = DINANet(user_num, item_num, hidden_dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.dina_net = self.dina_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.dina_net.parameters(), lr)

        accuracys, rmses, aucs, f1s = [], [], [], []

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, knowledge, response, diff = batch_data
                mean_diff = torch.mean(diff)
                diff = mean_diff.expand(diff.size())
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge: torch.Tensor = knowledge.to(device)
                diff: torch.Tensor = diff.to(device)
                predicted_response: torch.Tensor = self.dina_net(user_id, item_id, knowledge, diff)
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
                accuracys.append(accuracy)
                rmses.append(rmse)
                aucs.append(auc)
                f1s.append(f1)

                tra = pd.read_csv("../../../data/a0910/train.csv")
                us = list(set(list(tra['user_id'])))
                with open('u_f.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    for i in us:
                        ssf = self.dina_net.theta(torch.tensor(i)) >= 0
                        ssf = ssf.tolist()
                        sf = []
                        for j in ssf:
                            if str(j) == 'True':
                                sf.append(1)
                            else:
                                sf.append(0)
                        a = [i] + sf
                        writer.writerow(a)


        x = range(epoch)  # x

        plt.plot(x, accuracys, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
        ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
        plt.plot(x, rmses, 'g*--', alpha=0.5, linewidth=1, label='rmes')
        plt.plot(x, aucs, 'r*--', alpha=0.5, linewidth=1, label='auc')
        plt.plot(x, f1s, 'y*--', alpha=0.5, linewidth=1, label='f1')

        plt.legend()  # 显示上面的label
        plt.xlabel('epoch')  # x_label
        plt.ylabel('value')  # y_label

        # plt.ylim(-1,1)#仅设置y轴坐标范围
        plt.show()


    def eval(self, test_data, device="cpu") -> tuple:
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()
        y_pred = []
        y_true = []
        pred_class_all = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, knowledge, response, diff = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge: torch.Tensor = knowledge.to(device)
            diff: torch.Tensor = diff.to(device)
            pred: torch.Tensor = self.dina_net(user_id, item_id, knowledge, diff)
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

        self.dina_net.train()
        return accuracy, precision, rmse, recall, auc, f1

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
