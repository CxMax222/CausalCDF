import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from EduCDM import CDM

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#rd.seed(seed)

class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, knowledge_n, exer_n, student_n ):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, 1)
        self.k_difficulty = nn.Embedding(self.exer_n, 1)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        #self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        #self.drop_1 = nn.Dropout(p=0.5)
        #self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        #self.drop_2 = nn.Dropout(p=0.5)
        #self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) #* 1
        output = torch.sigmoid(input_x)

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.e_discrimination.apply(clipper)


    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)



class NCDM_IRT(CDM):
    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM_IRT, self).__init__()
        self.ncdm_irt_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.001, silence=False):
        self.ncdm_irt_net = self.ncdm_irt_net.to(device)
        self.ncdm_irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_irt_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id = torch.LongTensor([int(i[0]) for i in batch_data])
                item_id = torch.LongTensor([int(i[1]) for i in batch_data])
                knowledge_emb = torch.Tensor([list(i[2]) for i in batch_data])
                y = torch.Tensor([float(i[3]) for i in batch_data])
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb  = knowledge_emb.to(device)
                y  = y.to(device)
                pred: torch.Tensor = self.ncdm_irt_net(user_id, item_id)
                loss = loss_function(pred.squeeze(-1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.ncdm_irt_net.apply_clipper()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                accuracy, precision, rmse, recall, auc, f1 = self.eval(test_data, device=device)
                print('epoch= %d, accuracy= %f, precision=%f, rmse= %f, recall= %f, auc= %f, f1= %f' % (
                    epoch_i, accuracy, precision, rmse, recall, auc, f1))

    def eval(self, test_data, device="cpu"):
        self.ncdm_irt_net = self.ncdm_irt_net.to(device)
        self.ncdm_irt_net.eval()
        y_true, y_pred, pred_class_all = [], [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id = torch.LongTensor([int(i[0]) for i in batch_data])
            item_id = torch.LongTensor([int(i[1]) for i in batch_data])
            knowledge_emb = torch.Tensor([list(i[2]) for i in batch_data])
            y = torch.Tensor([float(i[3]) for i in batch_data])
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)
            pred: torch.Tensor = self.ncdm_irt_net(user_id, item_id)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

            batch_pred_class = []
            batch_ys = y
            batch_pred = pred
            for i in range(len(batch_ys)):
                if batch_pred[i] >= 0.5:
                    batch_pred_class.append(1)
                else:
                    batch_pred_class.append(0)
            pred_class_all += batch_pred_class

        y_pred = np.array(y_pred)
        y_pred = y_pred.astype(np.float32)
        y_true = np.array(y_true)
        y_true = y_true.astype(np.float32)
        pred_class_all = np.array(pred_class_all)

        accuracy = accuracy_score(y_true, pred_class_all)
        precision = precision_score(y_true, pred_class_all)
        recall = recall_score(y_true, pred_class_all)
        f1 = f1_score(y_true, pred_class_all)
        # compute RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        # compute AUC
        auc = roc_auc_score(y_true, y_pred)

        return accuracy, precision, rmse, recall, auc, f1

    def save(self, filepath):
        torch.save(self.ncdm_irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_irt_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)