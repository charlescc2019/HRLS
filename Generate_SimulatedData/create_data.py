from mxnet import autograd, nd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(1)

num_inputs = 1
num_examples = 500
true_kAA = [10.0]
true_bAA = 1.0
true_kAB = [-5.0]
true_bAB = -2
# 取期望为0 scale越大正太分布约矮越胖 scale越小正太分布越高越瘦

X_AA = np.random.uniform(0, 200, size=(num_examples, num_inputs))
Y_AA = true_kAA[0] * X_AA[:, 0] + true_bAA
Y_AA += np.random.normal(scale=80, size=Y_AA.shape)

plt.scatter(X_AA[:, 0], Y_AA, 1, 'b')
plt.plot(X_AA[:, 0], true_kAA * X_AA[:, 0] + true_bAA, color='red')
plt.savefig("AA")
plt.show()

X_AB = np.random.uniform(-200, 0, size=(num_examples, num_inputs))
Y_AB = true_kAB[0] * X_AB[:, 0] + true_bAB
Y_AB += np.random.normal(scale=50, size=Y_AB.shape)

plt.scatter(X_AB[:, 0], Y_AB, 1, 'b')
plt.plot(X_AB[:, 0], true_kAB * X_AB[:, 0] + true_bAB, color='red')
plt.savefig("AB")
plt.show()


data_AA = []
data_AB = []
# for i in range(10):
#     print(Y[i])

columns = ["X", "Y"]
for i in range(500):
    data_AA.append([X_AA[i][0], Y_AA[i]])
for i in range(500):
    data_AB.append([X_AB[i][0], Y_AB[i]])

dtAA = pd.DataFrame(data_AA, columns=columns)
dtAB = pd.DataFrame(data_AB, columns=columns)
dtAA.to_excel("manual_dataAA.xlsx")
dtAB.to_excel("manual_dataAB.xlsx")
