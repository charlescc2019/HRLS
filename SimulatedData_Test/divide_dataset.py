import pandas as pd
import random
import numpy

df_AA = pd.read_excel('dataset/manual_dataAA.xlsx')
df_AB = pd.read_excel('dataset/manual_dataAB.xlsx')

X_AA = df_AA['X']
Y_AA = df_AA['Y']

X_AB = df_AB['X']
Y_AB = df_AB['Y']

XAA = []
XAB = []
Y = []

A_train = []
AA_train = []
AB_train = []

A_test = []
AA_test = []
AB_test = []

random.seed(888)  # 设置随机种子以便实验复现

for i in range(len(X_AA)):

    if i >= 100:
        AA_train.append([X_AA[i], Y_AA[i]])
    else:
        AA_test.append([X_AA[i], Y_AA[i]])

for i in range(len(X_AB)):
    if i >= 100:
        AB_train.append([X_AB[i], Y_AB[i]])
    else:
        AB_test.append([X_AB[i], Y_AB[i]])
# print(min(len(AA), len(AB)))
for i in range(min(len(X_AA), len(X_AB))):
    XAA.append(X_AA[i])
    XAB.append(X_AB[i])
    Y.append(Y_AA[i] + Y_AB[i])
for i in range(len(XAA)):
    if i >= 100:
        A_train.append([XAA[i], XAB[i], Y[i]])
    else:
        A_test.append([XAA[i], XAB[i], Y[i]])

columns = ["X", "Y"]
colA = ["XAA", "XAB", "Y"]
dt_aa = pd.DataFrame(AA_test, columns=columns)
dt_ab = pd.DataFrame(AB_test, columns=columns)
dt_a = pd.DataFrame(A_test, columns=colA)

dt_aa_train = pd.DataFrame(AA_train, columns=columns)
dt_ab_train = pd.DataFrame(AB_train, columns=columns)
dt_a_train = pd.DataFrame(A_train, columns=colA)

dt_aa.to_excel("dataset/testdata_AA.xlsx")
dt_ab.to_excel("dataset/testdata_AB.xlsx")
dt_a.to_excel("dataset/testdata_A.xlsx")

dt_aa_train.to_excel("dataset/traindata_AA.xlsx")
dt_ab_train.to_excel("dataset/traindata_AB.xlsx")
dt_a_train.to_excel("dataset/traindata_A.xlsx")
