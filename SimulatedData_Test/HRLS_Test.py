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


def consistence_error(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, B3, K3, xAB1):
    ce = 0
    m = len(yA)
    for i in range(m):
        ce += abs(KA1 * xA1[i] + KA2 * xA2[i] + B1 - (K2 * xAA1[i] + B2 + K3 * xAB1[i] + B3))
    return ce/m


def HL_LOSS(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB):
    # add some consistence into the former loss function
    LOSS = 0

    m = len(yA)
    for i1 in range(m):
        LOSS += (yA[i1] - (KA1 * xA1[i1] + KA2 * xA2[i1] + B1)) ** 2 + (
                yAA[i1] - (K2 * xAA1[i1] + B2)) ** 2 + (yAB[i1] - (K3 * xAB1[i1] + B3)) ** 2 + \
                ((KA1 * xA1[i1] + KA2 * xA2[i1] + B1) - yA[i1]) ** 2

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


def gd_for_HL(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB, A1, Epochs):
    difference = float('inf')
    h_loss = []
    h_epoch = []
    m = len(yA)  # very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # pre_computing
    pre_hl_loss = HL_LOSS(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB)
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
            bA_grad += ((KA1 * xA1[i1] + KA2 * xA2[i1] + B1) - yA[i1] + KA1 * xA1[i1] + KA2 * xA2[i1] + B1 - B2 - K2 *
                        xAA1[i1] - B3 - K3 * xAB1[i1]) / m

            kA1_grad += ((KA1 * xA1[i1] + KA2 * xA2[i1] + B1) - yA[i1] + (
                    KA1 * xA1[i1] + KA2 * xA2[i1] + B1 - B2 - K2 * xAA1[i1] - B3 - K3 * xAB1[i1])) * xA1[i1] / m

            kA2_grad += ((KA1 * xA1[i1] + KA2 * xA2[i1] + B1) - yA[i1] + (
                    KA1 * xA1[i1] + KA2 * xA2[i1] + B1 - B2 - K2 * xAA1[i1] - B3 - K3 * xAB1[i1])) * xA2[i1] / m

            bAA_grad += ((K2 * xAA1[i1] + B2) - yAA[i1] - B1 - KA1 * xA1[i1] - KA2 * xA2[i1] +
                         B2 + K2 * xAA1[i1] + B3 + K3 * xAB1[i1]) / m

            kAA_grad += ((K2 * xAA1[i1] + B2) - yAA[i1] - B1 - KA1 * xA1[i1] - KA2 * xA2[i1] +
                         B2 + K2 * xAA1[i1] + B3 + K3 * xAB1[i1]) * xAA1[i1] / m

            bAB_grad += ((K3 * xAB1[i1] + B3) - yAB[i1] - B1 - KA1 * xA1[i1] - KA2 * xA2[i1] +
                         B2 + K2 * xAA1[i1] + B3 + K3 * xAB1[i1]) / m

            kAB_grad += ((K3 * xAB1[i1] + B3) - yAB[i1] - B1 - KA1 * xA1[i1] - KA2 * xA2[i1] +
                         B2 + K2 * xAA1[i1] + B3 + K3 * xAB1[i1]) * xAB1[i1] / m

        # First calculate the gradient,
        # then update the intercept and slope synchronously
        B1 = B1 - (A1 * bA_grad)
        KA1 = KA1 - (A1 * kA1_grad)
        KA2 = KA2 - (A1 * kA2_grad)
        B2 = B2 - (A1 * bAA_grad)
        K2 = K2 - (A1 * kAA_grad)
        B3 = B3 - (A1 * bAB_grad)
        K3 = K3 - (A1 * kAB_grad)

        hl_loss = HL_LOSS(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB)
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
regrA = LinearRegression()  # 二元线性回归
regrAA = LinearRegression()  # 一元线性回归
regrAB = LinearRegression()  # 一元线性回归

dfA = pandas.read_excel('dataset/traindata_A.xlsx')
XA = dfA[['XAA', 'XAB']]
XA1 = dfA['XAA']
XA2 = dfA['XAB']
YA = dfA['Y']
regrA.fit(XA, YA)
print('A的各系数' + str(regrA.coef_))
print('A的常数项：' + str(regrA.intercept_))
X2 = sm.add_constant(XA)
est = sm.OLS(YA, X2).fit()
print(est.summary())

dfAA = pandas.read_excel('dataset/traindata_AA.xlsx')
XAA = dfAA[['X']]
XAA1 = dfAA['X']
YAA = dfAA['Y']
regrAA.fit(XAA, YAA)
print('AA的各系数' + str(regrAA.coef_))
print('AA的常数项：' + str(regrAA.intercept_))
X2 = sm.add_constant(XAA)
estAA = sm.OLS(YAA, X2).fit()
print(estAA.summary())

dfAB = pandas.read_excel('dataset/traindata_AB.xlsx')
XAB = dfAB[['X']]
XAB1 = dfAB['X']
YAB = dfAB['Y']
regrAB.fit(XAB, YAB)
print('AB的各系数' + str(regrAB.coef_))
print('AB的常数项：' + str(regrAB.intercept_))
X2 = sm.add_constant(XAB)
estAB = sm.OLS(YAB, X2).fit()
print(estAB.summary())

a = 0.00001
# a = 0.000005
epochs = 2000
# epochs = 6000
b1, ka1, ka2, b2, k2, b3, k3 = 0, 0, 0, 0, 0, 0, 0
b1, ka1, ka2, b2, k2, b3, k3 = gd_for_HL(b1, ka1, ka2, XA1, XA2, YA, b2, k2, XAA1, YAA, b3, k3, XAB1, YAB, a, epochs)
print(b1, ka1, ka2, b2, k2, b3, k3)

# test
dfA_test = pandas.read_excel('dataset/testdata_A.xlsx')
dfAA_test = pandas.read_excel('dataset/testdata_AA.xlsx')
dfAB_test = pandas.read_excel('dataset/testdata_AB.xlsx')
XA1_T = dfA_test['XAA']
XA2_T = dfA_test['XAB']
YA_T = dfA_test['Y']
XAA1_T = dfAA_test['X']
YAA_T = dfAA_test['Y']
XAB1_T = dfAB_test['X']
YAB_T = dfAB_test['Y']
ceh = consistence_error(b1, ka1, ka2, XA1_T, XA2_T, YA_T, b2, k2, XAA1_T, b3, k3, XAB1_T)
ces = consistence_error(regrA.intercept_, regrA.coef_[0], regrA.coef_[1], XA1_T, XA2_T, YA_T,
                        regrAA.intercept_, regrAA.coef_[0], XAA1_T, regrAB.intercept_, regrAB.coef_[0], XAB1_T)

yA_predict_Th = []
yAA_predict_Th = []
yAB_predict_Th = []
yA_predict_Ts = []
yAA_predict_Ts = []
yAB_predict_Ts = []
yA_predict_Tbu = []
for i in range(len(YA_T)):
    yA_predict_Th.append(ka1 * XA1_T[i] + ka2 * XA2_T[i] + b1)
    yAA_predict_Th.append(k2 * XAA1_T[i] + b2)
    yAB_predict_Th.append(k3 * XAB1_T[i] + b3)
    yA_predict_Ts.append(regrA.coef_[0] * XA1_T[i] + regrA.coef_[1] * XA2_T[i] + b1)
    yAA_predict_Ts.append(regrAA.coef_[0] * XAA1_T[i] + b2)
    yAB_predict_Ts.append(regrAB.coef_[0] * XAB1_T[i] + b3)
    yA_predict_Tbu.append(regrAA.coef_[0] * XAA1_T[i] + b2 + regrAB.coef_[0] * XAB1_T[i] + b3)

# print(yA_predict_Ts)
# print(yA_predict_Tbu)

# top-down approach
yAA_predict_Ttd = []
yAB_predict_Ttd = []
print(yA_predict_Ts)
print(type(XAA1))
print(XAB1[0])


def proportion(YA, YAA, YAB):
    p_aa = 0
    p_ab = 0
    for i in range(len(YAA)):
        p_aa += YAA[i] / YA[i]
        p_ab += YAB[i] / YA[i]
    return p_aa / len(YAA), p_ab / len(YAA)


paa, pab = proportion(YA, YAA, YAB)
print(paa,pab)
for i in range(len(yA_predict_Ts)):
    yAA_predict_Ttd.append(paa * yA_predict_Ts[i])
    yAB_predict_Ttd.append(pab * yA_predict_Ts[i])

print('MAE for hierarchy:(A AA AB)')
print(mean_absolute_error(YA_T, yA_predict_Th))
print(mean_absolute_error(YAA_T, yAA_predict_Th))
print(mean_absolute_error(YAB_T, yAB_predict_Th))
print('MAE for separate:(A AA AB)')
print(mean_absolute_error(YA_T, yA_predict_Ts))
print(mean_absolute_error(YAA_T, yAA_predict_Ts))
print(mean_absolute_error(YAB_T, yAB_predict_Ts))
print('MAE for bottom-up:(A AA AB)')
print(mean_absolute_error(YA_T, yA_predict_Tbu))
print(mean_absolute_error(YAA_T, yAA_predict_Ts))
print(mean_absolute_error(YAB_T, yAB_predict_Ts))
print('MAE for top-down:(A AA AB)')
print(mean_absolute_error(YA_T, yA_predict_Ts))
print(mean_absolute_error(YAA_T, yAA_predict_Ttd))
print(mean_absolute_error(YAB_T, yAB_predict_Ttd))

print('For hierarchy liner model consistence error CE={0}'.format(ceh))
print('For separate liner model consistence error CE={0}'.format(ces))
cebu = 0
for i in range(len(yA_predict_Tbu)):
    cebu += abs(yA_predict_Tbu[i] - yAB_predict_Ts[i] - yAA_predict_Ts[i])
cebu = cebu / len(yA_predict_Tbu)
print('For bottom-up liner model consistence error CE={0}'.format(cebu))
cetd = 0
for i in range(len(yA_predict_Ts)):
    cetd += abs(yA_predict_Ts[i] - yAB_predict_Ttd[i] - yAA_predict_Ttd[i])
cetd = cetd / len(yAA_predict_Ttd)
print('For top-down liner model consistence error CE={0}'.format(cetd))

plt.figure()
ax = plt.axes(projection="3d")
X1 = dfA['XAA']
X2 = dfA['XAB']

X1, X2 = np.meshgrid(X1, X2)  # 2生成绘制3D图形所需的网络数据
Y = regrA.coef_[0] * X1 + regrA.coef_[1] * X2 + regrA.intercept_
surf = ax.plot_surface(X1, X2, Y, cmap=cm.Blues,
                       linewidth=0, antialiased=False)

ax.scatter(XA1, XA2, YA, c='r', marker='o', s=5)
ax.set_title('A-dataset')
ax.set_xlabel(r'$x_1$', fontsize=10, color='blue')
ax.set_ylabel(r'$x_2$', fontsize=10, color='blue')
ax.set_zlabel(r'$y$', fontsize=10, color='blue')
plt.show()
