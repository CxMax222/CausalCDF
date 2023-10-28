# coding: utf-8



import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#rd.seed(seed)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            #print(w)
            a = torch.relu(torch.neg(w))
            w.add_(a)

def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, a_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range
        self.out = nn.Linear(1, 1)

        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user, item, diff):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        #print(a)
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        scores = self.irf(theta, a, b, **self.irf_kwargs)
        #_scores = nn.ELU()(scores) + 1
        scores_with_diff = torch.mul(scores.unsqueeze(1), diff.unsqueeze(1))
        out = self.out(scores_with_diff)
        #print(self.out.state_dict())
        out = torch.squeeze(torch.sigmoid(out), dim=-1)
        return out


    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.a.apply(clipper)


class MIRT(CDM):
    def __init__(self, user_num, item_num, latent_dim, a_range=None):
        super(MIRT, self).__init__()
        self.irt_net = MIRTNet(user_num, item_num, latent_dim, a_range)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_function = nn.BCELoss()


        accuracys, rmses, aucs, f1s = [], [], [], []
        for e in range(epoch):
            losses = []
            print(lr)
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
                user_id, item_id, response, diff = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                diff: torch.Tensor = diff.to(device)
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id, diff)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                self.irt_net.apply_clipper()

                losses.append(loss.mean().item())
            #lr = lr *0.9
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                accuracy, precision, rmse, recall, auc, f1 = self.eval(test_data, device=device)
                print('epoch= %d, accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    e, accuracy, precision, rmse, recall, auc, f1))

                accuracys.append(accuracy)
                rmses.append(rmse)
                aucs.append(auc)
                f1s.append(f1)

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
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        pred_class_all = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response, diff = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            diff: torch.Tensor = diff.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id, diff)
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

        self.irt_net.train()
        return accuracy, precision, rmse, recall, auc, f1

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
