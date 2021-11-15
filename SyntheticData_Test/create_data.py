from mxnet import autograd, nd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# set the random seed as 1

np.random.seed(1)
num_inputs = 1
num_examples = 500
true_kP1 = [10.0]
true_bP1 = 1.0
true_kP2 = [-5.0]
true_bP2 = -2
# Take the expectation of 0. The larger the scale, the fatter the normal distribution is.
# The smaller the scale, the taller and thinner the normal distribution is

P1_X = np.random.uniform(0, 200, size=(num_examples, num_inputs))
P1_Y = true_kP1[0] * P1_X[:, 0] + true_bP1
P1_Y += np.random.normal(scale=80, size=P1_Y.shape)
# print(P1_Y)
plt.scatter(P1_X[:, 0], P1_Y, 1, 'b')
plt.plot(P1_X[:, 0], true_kP1 * P1_X[:, 0] + true_bP1, color='red')
plt.savefig("pic/P1")
plt.show()

P2_X = np.random.uniform(-200, 0, size=(num_examples, num_inputs))
P2_Y = true_kP2[0] * P2_X[:, 0] + true_bP2
P2_Y += np.random.normal(scale=50, size=P2_Y.shape)

plt.scatter(P2_X[:, 0], P2_Y, 1, 'b')
plt.plot(P2_X[:, 0], true_kP2 * P2_X[:, 0] + true_bP2, color='red')
plt.savefig("pic/P2")
plt.show()


data_P1 = []
data_P2 = []


columns = ["X", "Y"]
for i in range(500):
    data_P1.append([P1_X[i][0], P1_Y[i]])
for i in range(500):
    data_P2.append([P2_X[i][0], P2_Y[i]])

print(data_P1)
dtP1 = pd.DataFrame(data_P1, columns=columns)
dtP2 = pd.DataFrame(data_P2, columns=columns)
dtP1.to_excel("dataset/manual_dataP1.xlsx")
dtP2.to_excel("dataset/manual_dataP2.xlsx")
