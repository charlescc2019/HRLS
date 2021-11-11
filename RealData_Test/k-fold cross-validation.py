import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

print(datetime.datetime.now())

df = pd.read_excel('IncomeOfAutomobileIndustry.xlsx')
out = df['salary']
inp = df.drop(['salary'], axis=1)

# divide dataset into two sets
xAA, xAB, yAA, yAB = train_test_split(inp, out, train_size=0.5, shuffle=True, random_state=0)

xAA_train, xAA_test, yAA_train, yAA_test = train_test_split(xAA, yAA, train_size=0.8, shuffle=True, random_state=0)
xAB_train, xAB_test, yAB_train, yAB_test = train_test_split(xAB, yAB, train_size=0.8, shuffle=True, random_state=0)
# print(xAA_train)
xAA_train = xAA_train.reset_index(drop=True)
yAA_train = yAA_train.reset_index(drop=True)
xAB_train = xAB_train.reset_index(drop=True)
yAB_train = yAB_train.reset_index(drop=True)
xAA_test = xAA_test.reset_index(drop=True)
yAA_test = yAA_test.reset_index(drop=True)
xAB_test = xAB_test.reset_index(drop=True)
yAB_test = yAB_test.reset_index(drop=True)
# print(xAA_train)
# print(yAA_train)

xA_train = pd.concat([xAA_train, xAB_train], axis=1, ignore_index=True)
yA_train = yAA_train + yAB_train
xA_test = pd.concat([xAA_test, xAB_test], axis=1, ignore_index=True)
yA_test = yAA_test + yAB_test
print(xA_train)
print(yA_train)

model_AA = LinearRegression()
model_AB = LinearRegression()
model_A = LinearRegression()
model_AA.fit(xAA_train, yAA_train)
model_AB.fit(xAB_train, yAB_train)
model_A.fit(xA_train, yA_train)

# check the result
# model result

yAA_test_pred = model_AA.predict(xAA_test)
yAB_test_pred = model_AB.predict(xAB_test)
yA_test_pred = model_A.predict(xA_test)

print('for separate line AA:')
# y = 1106.596406754123x + 7801.587480961037
print('Coefficient:\n', model_AA.coef_)
print('Interception:\n', model_AA.intercept_)
print("MAE :", metrics.mean_absolute_error(yAA_test, yAA_test_pred))

print('for separate line AB:')
# y = 1108.3543168x +  7547.182440477868
print('Coefficient:\n', model_AB.coef_)
print('Interception:\n', model_AB.intercept_)
print("MAE :", metrics.mean_absolute_error(yAB_test, yAB_test_pred))

print('for separate line A:')
# y = 1106.596406754123x + 7801.587480961037
print('Coefficient:\n', model_A.coef_)
print('Interception:\n', model_A.intercept_)
print("MAE :", metrics.mean_absolute_error(yA_test, yA_test_pred))


# construct the hierarchy model

def consistence_error(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, B3, K3, xAB1):
    ce = 0
    m = len(yA)
    for i in range(m):
        ce += abs(KA1 * xA1[i] + KA2 * xA2[i] + B1 - (K2 * xAA1[i] + B2 + K3 * xAB1[i] + B3))
    return ce


def HL_LOSS(B1, KA1, KA2, xA1, xA2, yA, B2, K2, xAA1, yAA, B3, K3, xAB1, yAB):
    # add some consistence into the former loss function
    LOSS, lossA, lossAA, lossAB = 0, 0, 0, 0

    m = len(yA)

    for i1 in range(m):
        LOSS += (yA[i1] - (KA1 * xA1[i1] + KA2 * xA2[i1] + B1)) ** 2 + (
                yAA[i1] - (K2 * xAA1[i1] + B2)) ** 2 + (yAB[i1] - (K3 * xAB1[i1] + B3)) ** 2 + \
                ((KA1 * xA1[i1] + KA2 * xA2[i1] + B1) - (K2 * xAA1[i1] + B2) - (K3 * xAB1[i1] + B3)) ** 2

    return LOSS / float(m) / 2.0


def proportion(YA, YAA, YAB):
    p_aa = 0
    p_ab = 0
    for i in range(len(YAA)):
        p_aa += YAA[i] / YA[i]
        p_ab += YAB[i] / YA[i]
    return p_aa / len(YAA), p_ab / len(YAA)


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
            # print('error = {0}'.format(hl_loss))
            if hl_loss > pre_hl_loss:
                A1 = A1 / 2
            else:
                difference = pre_hl_loss - hl_loss
            pre_hl_loss = hl_loss
            if difference < 0.0000001:
                break

    plt.plot(h_epoch, h_loss)
    plt.title('LOSS curve for HL approach')
    plt.xlabel('h_epoch')
    plt.ylabel('h_loss')
    plt.show()
    return B1, KA1, KA2, B2, K2, B3, K3


a = 0.1
# a = 0.000005
epochs = 5000
# epochs = 6000
b1, ka1, ka2, b2, k2, b3, k3 = 0, 0, 0, 0, 0, 0, 0
# b1, ka1, ka2, b2, k2, b3, k3 = gd_for_HL(b1, ka1, ka2, xA_train[0], xA_train[1], yA_train, b2, k2, xAA_train['years'],
#                                          yAA_train, b3, k3, xAB_train['years'], yAB_train, a, epochs)
# print(b1, ka1, ka2, b2, k2, b3, k3)
# hierarchy
yA_predict_Th = []
yAA_predict_Th = []
yAB_predict_Th = []
for i in range(len(yA_test)):
    yA_predict_Th.append(ka1 * xA_test[0][i] + ka2 * xA_test[1][i] + b1)
    yAA_predict_Th.append(k2 * xAA_test['years'][i] + b2)
    yAB_predict_Th.append(k3 * xAB_test['years'][i] + b3)

print('MAE for hierarchy:(AA AB A)')

print(metrics.mean_absolute_error(yAA_test, yAA_predict_Th))
print(metrics.mean_absolute_error(yAB_test, yAB_predict_Th))
print(metrics.mean_absolute_error(yA_test, yA_predict_Th))

ceh = consistence_error(b1, ka1, ka2, xA_test[0], xA_test[1], yA_test, b2, k2, xAA_test['years'],
                        b3, k3, xAB_test['years'])
ces = consistence_error(model_A.intercept_, model_A.coef_[0], model_A.coef_[1], xA_test[0], xA_test[1], yA_test,
                        model_AA.intercept_, model_AA.coef_[0], xAA_test['years'], model_AB.intercept_,
                        model_AB.coef_[0],
                        xAB_test['years'])
print('For hierarchy liner model consistence error CE={0}'.format(ceh))
print('For separate liner model consistence error CE={0}'.format(ces))
print('\n')

# 五折交叉验证
k = 5
num_validation_samples = len(yAA) // k
xAA = xAA.reset_index(drop=True)
yAA = yAA.reset_index(drop=True)
xAB = xAB.reset_index(drop=True)
yAB = yAB.reset_index(drop=True)
xA = pd.concat([xAA, xAB], axis=1, ignore_index=True)
yA = yAA + yAB
MAE_a_S = []
MAE_aa_S = []
MAE_ab_S = []
MAE_a_h = []
MAE_aa_h = []
MAE_ab_h = []
MAE_a_bu = []
MAE_aa_bu = []
MAE_ab_bu = []
MAE_a_td = []
MAE_aa_td = []
MAE_ab_td = []
ce_s = []
ce_h = []
ce_bu = []
ce_td = []
for fold in range(k):

    testing_data_xaa = xAA[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_yaa = yAA[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_xaa = pd.concat([xAA[:num_validation_samples * fold], xAA[num_validation_samples * (fold + 1):]])
    training_data_yaa = pd.concat([yAA[:num_validation_samples * fold], yAA[num_validation_samples * (fold + 1):]])

    testing_data_xab = xAB[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_yab = yAB[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_xab = pd.concat([xAB[:num_validation_samples * fold], xAB[num_validation_samples * (fold + 1):]])
    training_data_yab = pd.concat([yAB[:num_validation_samples * fold], yAB[num_validation_samples * (fold + 1):]])

    testing_data_xa = xA[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_ya = yA[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_xa = pd.concat([xA[:num_validation_samples * fold], xA[num_validation_samples * (fold + 1):]])
    training_data_ya = pd.concat([yA[:num_validation_samples * fold], yA[num_validation_samples * (fold + 1):]])

    testing_data_xaa = testing_data_xaa.reset_index(drop=True)
    testing_data_yaa = testing_data_yaa.reset_index(drop=True)
    training_data_xaa = training_data_xaa.reset_index(drop=True)
    training_data_yaa = training_data_yaa.reset_index(drop=True)

    testing_data_xab = testing_data_xab.reset_index(drop=True)
    testing_data_yab = testing_data_yab.reset_index(drop=True)
    training_data_xab = training_data_xab.reset_index(drop=True)
    training_data_yab = training_data_yab.reset_index(drop=True)

    testing_data_xa = testing_data_xa.reset_index(drop=True)
    testing_data_ya = testing_data_ya.reset_index(drop=True)
    training_data_xa = training_data_xa.reset_index(drop=True)
    training_data_ya = training_data_ya.reset_index(drop=True)
    a = 0.1
    # a = 0.000005
    epochs = 5000
    b1, ka1, ka2, b2, k2, b3, k3 = 0, 0, 0, 0, 0, 0, 0
    b1, ka1, ka2, b2, k2, b3, k3 = gd_for_HL(b1, ka1, ka2,
                                             training_data_xa[0],
                                             training_data_xa[1],
                                             training_data_ya, b2, k2,
                                             training_data_xaa['years'],
                                             training_data_yaa, b3, k3,
                                             training_data_xab['years'],
                                             training_data_yab, a, epochs)
    # print(b1, ka1, ka2, b2, k2, b3, k3)
    model_AA.fit(training_data_xaa, training_data_yaa)
    model_AB.fit(training_data_xab, training_data_yab)
    model_A.fit(training_data_xa, training_data_ya)
    yAA_test_pred = model_AA.predict(testing_data_xaa)  # 返回的是list
    yAB_test_pred = model_AB.predict(testing_data_xab)
    yA_test_pred = model_A.predict(testing_data_xa)
    #     # Bottom Up
    yA_test_pred_bu = []
    for i in range(len(yAA_test_pred)):
        yA_test_pred_bu.append(yAA_test_pred[i] + yAB_test_pred[i])
    print('MAE for BU model:(AA AB A)')
    m1 = metrics.mean_absolute_error(testing_data_yaa, yAA_test_pred)
    m2 = metrics.mean_absolute_error(testing_data_yab, yAB_test_pred)
    m3 = metrics.mean_absolute_error(testing_data_ya, yA_test_pred_bu)
    print(m1)
    print(m2)
    print(m3)
    MAE_a_bu.append(m3)
    MAE_ab_bu.append(m2)
    MAE_aa_bu.append(m1)
    # top down
    yAA_test_pred_td = []
    yAB_test_pred_td = []
    paa, pab = proportion(training_data_ya, training_data_yaa, training_data_yab)

    print(training_data_yab)
    for i in range(len(testing_data_ya)):
        yAA_test_pred_td.append(paa * yA_test_pred[i])
        yAB_test_pred_td.append(pab * yA_test_pred[i])
    print('MAE for top down:(AA AB A)')
    m1 = metrics.mean_absolute_error(testing_data_yaa, yAA_test_pred_td)
    m2 = metrics.mean_absolute_error(testing_data_yab, yAB_test_pred_td)
    m3 = metrics.mean_absolute_error(testing_data_ya, yA_test_pred)
    print(m1)
    print(m2)
    print(m3)
    MAE_a_td.append(m3)
    MAE_ab_td.append(m2)
    MAE_aa_td.append(m1)

    # base model
    print('MAE for separate:(AA AB A)')
    m1 = metrics.mean_absolute_error(testing_data_yaa, yAA_test_pred)
    m2 = metrics.mean_absolute_error(testing_data_yab, yAB_test_pred)
    m3 = metrics.mean_absolute_error(testing_data_ya, yA_test_pred)

    print(m1)
    print(m2)
    print(m3)
    MAE_a_S.append(m3)
    MAE_ab_S.append(m2)
    MAE_aa_S.append(m1)

    # hierarchy model

    yA_predict = []
    yAA_predict = []
    yAB_predict = []
    for i in range(len(testing_data_ya)):
        yA_predict.append(ka1 * testing_data_xa[0][i] + ka2 * testing_data_xa[1][i] + b1)
        yAA_predict.append(k2 * testing_data_xaa['years'][i] + b2)
        yAB_predict.append(k3 * testing_data_xab['years'][i] + b3)
    print('MAE for hierarchy:(AA AB A)')
    m1 = metrics.mean_absolute_error(testing_data_yaa, yAA_predict)
    m2 = metrics.mean_absolute_error(testing_data_yab, yAB_predict)
    m3 = metrics.mean_absolute_error(testing_data_ya, yA_predict)
    MAE_a_h.append(m3)
    MAE_ab_h.append(m2)
    MAE_aa_h.append(m1)

    print(m1)
    print(m2)
    print(m3)

    # 开始计算consistency error

    ceh = 0
    for i in range(len(yAA_predict)):
        ceh += abs(yA_predict[i] - yAA_predict[i] - yAB_predict[i])/len(yAA_predict)
    ce_h.append(ceh)
    ces = 0
    for i in range(len(yA_test_pred)):
        ces += abs(yA_test_pred[i] - yAA_test_pred[i] - yAB_test_pred[i])/len(yA_test_pred)
    ce_s.append(ces)
    # 11111111111111111111111111111111111111111111111111111111111111111111111111111
    cebu = 0
    for i in range(len(yA_test_pred_bu)):
        cebu += abs(yA_test_pred_bu[i] - yAA_test_pred[i] - yAB_test_pred[i])/len(yA_test_pred_bu)
    ce_bu.append(cebu)

    cetd = 0
    for i in range(len(yAA_test_pred_td)):
        cetd += abs(yA_test_pred[i] - yAA_test_pred_td[i] - yAB_test_pred_td[i])/len(yAA_test_pred_td)
    ce_td.append(cetd)

    # 11111111111111111111111111111111111111111111111111111111111111111111111111111
    print('For bottom up liner model consistence error CE={0}'.format(cebu))
    print('For top down liner model consistence error CE={0}'.format(cetd))
    print('For separate liner model consistence error CE={0}'.format(ces))
    print('For hierarchy liner model consistence error CE={0}'.format(ceh))

print('5-fold-cross-validation')

print('\n')
print('The average result for 5 fold cross validation : ')
print('MAE for separate model(AA,AB,A)')
print(sum(MAE_aa_S) / k)
print(sum(MAE_ab_S) / k)
print(sum(MAE_a_S) / k)
print('For separate liner model consistence error CE={0}'.format(sum(ce_s) / k))
print('MAE for hierarchy model:(AA,AB,A)')
print(sum(MAE_aa_h) / k)
print(sum(MAE_ab_h) / k)
print(sum(MAE_a_h) / k)
print('For hierarchy liner model consistence error CE={0}'.format(sum(ce_h) / k))
print('MAE for bottom-up model:(AA,AB,A)')
print(sum(MAE_aa_bu) / k)
print(sum(MAE_ab_bu) / k)
print(sum(MAE_a_bu) / k)
print('For bottom-up liner model consistence error CE={0}'.format(sum(ce_bu) / k))
print('MAE for top-down model:(AA,AB,A)')
print(sum(MAE_aa_td) / k)
print(sum(MAE_ab_td) / k)
print(sum(MAE_a_td) / k)
print('For top down liner model consistence error CE={0}'.format(sum(ce_td) / k))
