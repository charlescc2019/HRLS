import datetime
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import pandas
import warnings
from matplotlib import pyplot as plt, cm
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))  # 获取当前py文件的父目录
types = sys.getfilesystemencoding()
sys.stdout = Logger('log.txt')  # 输出文件
print(datetime.datetime.now())


def MCE(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, B3, K3, P2_x):
    ce = 0
    m = len(yA)
    for i in range(m):
        ce += abs(KA1 * P0_x1[i] + KA2 * P0_x2[i] + B1 - (K2 * P1_x[i] + B2 + K3 * P2_x[i] + B3))
    return ce/m


def HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, P0_y, B2, K2, P1_x, P1_y, B3, K3, P2_x, P2_y):
    # add some consistence into the former loss function
    LOSS = 0

    m = len(P0_y)
    for i1 in range(m):
        LOSS += (P0_y[i1] - (KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1)) ** 2 + (
                P1_y[i1] - (K2 * P1_x[i1] + B2)) ** 2 + (P2_y[i1] - (K3 * P2_x[i1] + B3)) ** 2 + \
                ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - P0_y[i1]) ** 2

    return LOSS / float(m) / 2.0


def SE_LOSS(b, k, x, y):
    loss = 0
    m = len(x)
    for i1 in range(m):
        loss += (y[i1] - (k * x[i1] + b)) ** 2
    return loss / float(m) / 2.0


def SEA_LOSS(b, K1, K2, x1, x2, y):
    loss = 0
    m = len(y)
    for i1 in range(m):
        loss += (y[i1] - (K1 * x1[i1] + K2 * x2[i1] + b)) ** 2
    return loss / float(m) / 2.0


def gd_for_HL(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, yAA, B3, K3, P2_x, P2_y, A1, Epochs):
    difference = float('inf')
    h_loss = []
    h_epoch = []
    m = len(yA)  # very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # pre_computing
    pre_hl_loss = HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, yAA, B3, K3, P2_x, P2_y)
    for item in range(Epochs):
        bA_grad = 0
        kA1_grad = 0
        kA2_grad = 0
        bAA_grad = 0
        kAA_grad = 0
        bAB_grad = 0
        kAB_grad = 0
        # Calculate the gradients of RSS by iterations
        for i1 in range(m):
            bA_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - yA[i1] + KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 *
                        P1_x[i1] - B3 - K3 * P2_x[i1]) / m

            kA1_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - yA[i1] + (
                    KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 * P1_x[i1] - B3 - K3 * P2_x[i1])) * P0_x1[i1] / m

            kA2_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - yA[i1] + (
                    KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 * P1_x[i1] - B3 - K3 * P2_x[i1])) * P0_x2[i1] / m

            bAA_grad += ((K2 * P1_x[i1] + B2) - yAA[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
                         B2 + K2 * P1_x[i1] + B3 + K3 * P2_x[i1]) / m

            kAA_grad += ((K2 * P1_x[i1] + B2) - yAA[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
                         B2 + K2 * P1_x[i1] + B3 + K3 * P2_x[i1]) * P1_x[i1] / m

            bAB_grad += ((K3 * P2_x[i1] + B3) - P2_y[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
                         B2 + K2 * P1_x[i1] + B3 + K3 * P2_x[i1]) / m

            kAB_grad += ((K3 * P2_x[i1] + B3) - P2_y[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
                         B2 + K2 * P1_x[i1] + B3 + K3 * P2_x[i1]) * P2_x[i1] / m

        # First calculate the gradient,
        # then update the intercept and slope synchronously
        B1 = B1 - (A1 * bA_grad)
        KA1 = KA1 - (A1 * kA1_grad)
        KA2 = KA2 - (A1 * kA2_grad)
        B2 = B2 - (A1 * bAA_grad)
        K2 = K2 - (A1 * kAA_grad)
        B3 = B3 - (A1 * bAB_grad)
        K3 = K3 - (A1 * kAB_grad)

        hl_loss = HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, yAA, B3, K3, P2_x, P2_y)
        if item % 10 == 0:
            h_epoch.append(item)
            h_loss.append(hl_loss)
            print('error = {0}'.format(hl_loss))
            if hl_loss > pre_hl_loss:
                A1 = A1 / 2
            else:
                difference = pre_hl_loss - hl_loss
            pre_hl_loss = hl_loss
            if difference < 0.0000001:
                break
        # 88原来的代码
        # if item % 10 == 0:
        #     hl_loss = HL_LOSS(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB)
        #     h_epoch.append(item)
        #     h_loss.append(hl_loss)
        #     print('error = {0}'.format(hl_loss))

    plt.plot(h_epoch, h_loss)
    plt.title('LOSS curve for HL approach')
    plt.xlabel('h_epoch')
    plt.ylabel('h_loss')
    plt.show()
    return B1, KA1, KA2, B2, K2, B3, K3


# training
warnings.filterwarnings("ignore")
model_P0 = LinearRegression()  # 二元线性回归
model_P1 = LinearRegression()  # 一元线性回归
model_P2 = LinearRegression()  # 一元线性回归

dfP0 = pandas.read_excel('dataset/traindata_P0.xlsx')
P0_X = dfP0[['p1_x', 'p2_x']]
P0_X1 = dfP0['p1_x']
P0_X2 = dfP0['p2_x']
P0_Y = dfP0['p0_y']
model_P0.fit(P0_X, P0_Y)
X2 = sm.add_constant(P0_X)
est = sm.OLS(P0_Y, X2).fit()
print(est.summary())

df_P1 = pandas.read_excel('dataset/traindata_P1.xlsx')
P1_X = df_P1[['X']]
P1_X1 = df_P1['X']
P1_Y = df_P1['Y']
model_P1.fit(P1_X, P1_Y)
X2 = sm.add_constant(P1_X)
estP1 = sm.OLS(P1_Y, X2).fit()
print(estP1.summary())

dfP2 = pandas.read_excel('dataset/traindata_P2.xlsx')
P2_X = dfP2[['X']]
P2_X1 = dfP2['X']
P2_Y = dfP2['Y']
model_P2.fit(P2_X, P2_Y)
X2 = sm.add_constant(P2_X)
estP2 = sm.OLS(P2_Y, X2).fit()
print(estP2.summary())

a = 0.00001
# a = 0.000005
epochs = 3000
# epochs = 6000
b1, ka1, ka2, b2, k2, b3, k3 = 0, 0, 0, 0, 0, 0, 0
b1, ka1, ka2, b2, k2, b3, k3 = gd_for_HL(b1, ka1, ka2, P0_X1, P0_X2, P0_Y, b2, k2, P1_X1, P1_Y, b3, k3, P2_X1,
                                         P2_Y, a, epochs)
print(b1, ka1, ka2, b2, k2, b3, k3)

# test
dfP0_test = pandas.read_excel('dataset/testdata_P0.xlsx')
dfP1_test = pandas.read_excel('dataset/testdata_P1.xlsx')
dfP2_test = pandas.read_excel('dataset/testdata_P2.xlsx')
P0_X1_T = dfP0_test['p1_x']
P0_X2_T = dfP0_test['p2_x']
P0_Y_T = dfP0_test['p0_y']
P1_X_T = dfP1_test['X']
P1_Y_T = dfP1_test['Y']
P2_X_T = dfP2_test['X']
P2_Y_T = dfP2_test['Y']
mceh = MCE(b1, ka1, ka2, P0_X1_T, P0_X2_T, P0_Y_T, b2, k2, P1_X_T, b3, k3, P2_X_T)
mces = MCE(model_P0.intercept_, model_P0.coef_[0], model_P0.coef_[1], P0_X1_T, P0_X2_T, P0_Y_T,
           model_P1.intercept_, model_P1.coef_[0], P1_X_T, model_P2.intercept_, model_P2.coef_[0], P2_X_T)

p0_y_predict_Th = []
p1_y_predict_Th = []
p2_y_predict_Th = []
p0_y_predict_Ts = []
p1_y_predict_Ts = []
p2_y_predict_Ts = []
p0_y_predict_Tbu = []
for i in range(len(P0_Y_T)):
    p0_y_predict_Th.append(ka1 * P0_X1_T[i] + ka2 * P0_X2_T[i] + b1)
    p1_y_predict_Th.append(k2 * P1_X_T[i] + b2)
    p2_y_predict_Th.append(k3 * P2_X_T[i] + b3)
    p0_y_predict_Ts.append(model_P0.coef_[0] * P0_X1_T[i] + model_P0.coef_[1] * P0_X2_T[i] + b1)
    p1_y_predict_Ts.append(model_P1.coef_[0] * P1_X_T[i] + b2)
    p2_y_predict_Ts.append(model_P2.coef_[0] * P2_X_T[i] + b3)
    p0_y_predict_Tbu.append(model_P1.coef_[0] * P1_X_T[i] + b2 + model_P2.coef_[0] * P2_X_T[i] + b3)

# print(yA_predict_Ts)
# print(yA_predict_Tbu)

# top-down approach
p1_y_predict_Ttd = []
p2_y_predict_Ttd = []
print(p0_y_predict_Ts)
print(type(P1_X1))
print(P2_X1[0])


def proportion(YA, YAA, YAB):
    p_aa = 0
    p_ab = 0
    for i in range(len(YAA)):
        p_aa += YAA[i] / YA[i]
        p_ab += YAB[i] / YA[i]
    return p_aa / len(YAA), p_ab / len(YAA)


proportion1, proportion2 = proportion(P0_Y, P1_Y, P2_Y)
print(proportion1, proportion2)
for i in range(len(p0_y_predict_Ts)):
    p1_y_predict_Ttd.append(proportion1 * p0_y_predict_Ts[i])
    p2_y_predict_Ttd.append(proportion2 * p0_y_predict_Ts[i])

print('MAE for hierarchy:(P0 P1 P2)')
print(mean_absolute_error(P0_Y_T, p0_y_predict_Th))
print(mean_absolute_error(P1_Y_T, p1_y_predict_Th))
print(mean_absolute_error(P2_Y_T, p2_y_predict_Th))
print('MAE for separate:(P0 P1 P2)')
print(mean_absolute_error(P0_Y_T, p0_y_predict_Ts))
print(mean_absolute_error(P1_Y_T, p1_y_predict_Ts))
print(mean_absolute_error(P2_Y_T, p2_y_predict_Ts))
print('MAE for bottom-up:(P0 P1 P2)')
print(mean_absolute_error(P0_Y_T, p0_y_predict_Tbu))
print(mean_absolute_error(P1_Y_T, p1_y_predict_Ts))
print(mean_absolute_error(P2_Y_T, p2_y_predict_Ts))
print('MAE for top-down:(P0 P1 P2)')
print(mean_absolute_error(P0_Y_T, p0_y_predict_Ts))
print(mean_absolute_error(P1_Y_T, p1_y_predict_Ttd))
print(mean_absolute_error(P2_Y_T, p2_y_predict_Ttd))

print('For HRLS model mean consistence error MCE={0}'.format(mceh))
print('For Separate model consistence error MCE={0}'.format(mces))
mcebu = 0
for i in range(len(p0_y_predict_Tbu)):
    mcebu += abs(p0_y_predict_Tbu[i] - p2_y_predict_Ts[i] - p1_y_predict_Ts[i])
mcebu = mcebu / len(p0_y_predict_Tbu)
print('For BU model mean consistence error MCE={0}'.format(mcebu))
mcetd = 0
for i in range(len(p0_y_predict_Ts)):
    mcetd += abs(p0_y_predict_Ts[i] - p2_y_predict_Ttd[i] - p1_y_predict_Ttd[i])
mcetd = mcetd / len(p1_y_predict_Ttd)
print('For TD model mean consistence error MCE={0}'.format(mcetd))

plt.figure()
ax = plt.axes(projection="3d")
X1 = dfP0['p1_x']
X2 = dfP0['p2_x']

X1, X2 = np.meshgrid(X1, X2)
Y = model_P0.coef_[0] * X1 + model_P0.coef_[1] * X2 + model_P0.intercept_
surf = ax.plot_surface(X1, X2, Y, cmap=cm.Blues,
                       linewidth=0, antialiased=False)

ax.scatter(P0_X1, P0_X2, P0_Y, c='r', marker='o', s=5)
ax.set_title('P0-dataset')
ax.set_xlabel(r'$x_1$', fontsize=10, color='blue')
ax.set_ylabel(r'$x_2$', fontsize=10, color='blue')
ax.set_zlabel(r'$y$', fontsize=10, color='blue')
plt.show()
