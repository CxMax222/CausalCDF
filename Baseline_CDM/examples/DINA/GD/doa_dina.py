import pandas as pd
import csv
from tqdm import tqdm






train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")
df_item = pd.read_csv("../../../data/a0910/item.csv")
students = list(set(list(train_data['user_id'])))
exers = list(set(list(train_data['item_id']) + list(valid_data['item_id']) + list(test_data['item_id'])))
e_k= {}
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    e_k[item_id] = knowledge_codes

k_e= {}
for k in range(1,124):
    k_e[k] = []
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    for i in knowledge_codes:
        k_e[i].append(item_id)



jab ={}

tr = dict(zip(tuple(zip(train_data['user_id'],train_data['item_id'])),train_data['score']))
v = dict(zip(tuple(zip(valid_data['user_id'],valid_data['item_id'])),valid_data['score']))
te = dict(zip(tuple(zip(test_data['user_id'],test_data['item_id'])),test_data['score']))
jab.update(tr)
jab.update(v)
jab.update(te)



s_f ={}
with open('u_f.csv', 'r',encoding='utf-8', newline='') as csvfile:
    cs= csv.reader(csvfile)
    cs = list(cs)
for i in cs:
    s_f[int(i[0])] = [float(j) for j in i[1:]]

DOAs = []
for k in tqdm(range(1,124)):

    sum_F = 0
    Z = 0
    for a in students:
        for b in students:
            if s_f[a][k - 1] > s_f[b][k - 1]:
                sum_2 = 0
                sum_1 = 0
                for j in k_e[k]:
                    if (a, j) in jab.keys() and (b, j) in jab.keys():
                        # sum_1 += 1
                        if jab[a, j] != jab[b, j]:
                            sum_1 += 1
                        if jab[a, j] > jab[b, j]:
                            sum_2 += 1
                if sum_1 != 0:
                    Z += 1
                    sum_F += sum_2 / sum_1

                            #print(sum_F)
    DOA = sum_F/Z
    print(DOA)
    DOAs.append(DOA)
with open('DOA.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in DOAs:
        writer.writerow([i])
print(sum(DOAs)/len(DOAs))
