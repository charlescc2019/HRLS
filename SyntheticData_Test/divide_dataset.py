import pandas as pd
import random
import numpy

df_P1 = pd.read_excel('dataset/manual_dataP1.xlsx')
df_P2 = pd.read_excel('dataset/manual_dataP2.xlsx')

P1_X = df_P1['X']
P1_Y = df_P1['Y']

P2_X = df_P2['X']
P2_Y = df_P2['Y']

p1_x = []
p2_x = []
p0_y = []

P0_train = []
P1_train = []
P2_train = []

P0_test = []
P1_test = []
P2_test = []

random.seed(888)  # 设置随机种子以便实验复现

for i in range(len(P1_X)):

    if i < 100:
        P1_test.append([P1_X[i], P1_Y[i]])
    else:
        P1_train.append([P1_X[i], P1_Y[i]])

for i in range(len(P2_X)):
    if i < 100:
        P2_test.append([P2_X[i], P2_Y[i]])
    else:
        P2_train.append([P2_X[i], P2_Y[i]])
# print(min(len(AA), len(AB)))
for i in range(min(len(P1_X), len(P2_X))):
    p1_x.append(P1_X[i])
    p2_x.append(P2_X[i])
    p0_y.append(P1_Y[i] + P2_Y[i])
for i in range(len(p1_x)):
    if i >= 100:
        P0_train.append([p1_x[i], p2_x[i], p0_y[i]])
    else:
        P0_test.append([p1_x[i], p2_x[i], p0_y[i]])

columns = ["X", "Y"]
colP0 = ["p1_x", "p2_x", "p0_y"]
dt_p1 = pd.DataFrame(P1_test, columns=columns)
dt_p2 = pd.DataFrame(P2_test, columns=columns)
dt_p0 = pd.DataFrame(P0_test, columns=colP0)

dt_p1_train = pd.DataFrame(P1_train, columns=columns)
dt_p2_train = pd.DataFrame(P2_train, columns=columns)
dt_p0_train = pd.DataFrame(P0_train, columns=colP0)

dt_p1.to_excel("dataset/testdata_P1.xlsx")
dt_p2.to_excel("dataset/testdata_P2.xlsx")
dt_p0.to_excel("dataset/testdata_P0.xlsx")

dt_p1_train.to_excel("dataset/traindata_P1.xlsx")
dt_p2_train.to_excel("dataset/traindata_P2.xlsx")
dt_p0_train.to_excel("dataset/traindata_P0.xlsx")
