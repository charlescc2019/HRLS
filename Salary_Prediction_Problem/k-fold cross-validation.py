import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

print(datetime.datetime.now())

df = pd.read_excel('dataset/IncomeOfAutomobileIndustry.xlsx')
out = df['salary']
inp = df.drop(['salary'], axis=1)

# divide dataset into two sets
# The random_state for train_test_split() is 0!
p1_x, p2_x, p1_y, p2_y = train_test_split(inp, out, train_size=0.5, shuffle=True, random_state=0)
p1_x_train, p1_x_test, p1_y_train, p1_y_test = train_test_split(p1_x, p1_y, train_size=0.8,
                                                                shuffle=True, random_state=0)
p2_x_train, p2_x_test, p2_y_train, p2_y_test = train_test_split(p2_x, p2_y, train_size=0.8, shuffle=True,
                                                                random_state=0)
p1_x_train = p1_x_train.reset_index(drop=True)
p1_y_train = p1_y_train.reset_index(drop=True)
p2_x_train = p2_x_train.reset_index(drop=True)
p2_y_train = p2_y_train.reset_index(drop=True)
p1_x_test = p1_x_test.reset_index(drop=True)
p1_y_test = p1_y_test.reset_index(drop=True)
p2_x_test = p2_x_test.reset_index(drop=True)
p2_y_test = p2_y_test.reset_index(drop=True)

p0_x_train = pd.concat([p1_x_train, p2_x_train], axis=1, ignore_index=True)
p0_y_train = p1_y_train + p2_y_train
p0_x_test = pd.concat([p1_x_test, p2_x_test], axis=1, ignore_index=True)
p0_y_test = p1_y_test + p2_y_test

model_p1 = LinearRegression()
model_p2 = LinearRegression()
model_p0 = LinearRegression()
model_p1.fit(p1_x_train, p1_y_train)
model_p2.fit(p2_x_train, p2_y_train)
model_p0.fit(p0_x_train, p0_y_train)

# check the result
# model result

p1_y_test_pred = model_p1.predict(p1_x_test)
p2_y_test_pred = model_p2.predict(p2_x_test)
p0_y_test_pred = model_p0.predict(p0_x_test)


# construct the hierarchy model
# Mean Consistency Error

def MCE(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, B3, K3, P2_x):
    ce = 0
    m = len(yA)
    for i in range(m):
        ce += abs(KA1 * P0_x1[i] + KA2 * P0_x2[i] + B1 - (K2 * P1_x[i] + B2 + K3 * P2_x[i] + B3))
    return ce / len(yA)


def HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, yA, B2, K2, P1_x, yAA, B3, K3, P2_x, P2_y):
    # add some consistence into the former loss function
    LOSS, lossA, lossAA, lossAB = 0, 0, 0, 0

    m = len(yA)

    for i1 in range(m):
        LOSS += (yA[i1] - (KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1)) ** 2 + (
                yAA[i1] - (K2 * P1_x[i1] + B2)) ** 2 + (P2_y[i1] - (K3 * P2_x[i1] + B3)) ** 2 + \
                ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - (K2 * P1_x[i1] + B2) - (K3 * P2_x[i1] + B3)) ** 2

    return LOSS / float(m) / 2.0


def proportion(P0_Y, P1_Y, P2_Y):
    p_aa = 0
    p_ab = 0
    for i in range(len(P1_Y)):
        p_aa += P1_Y[i] / P0_Y[i]
        p_ab += P2_Y[i] / P0_Y[i]
    return p_aa / len(P1_Y), p_ab / len(P1_Y)


def gd_for_HL(B1, KA1, KA2, P0_x1, P0_x2, P0_y, B2, K2, P1_x, P1_y, B3, K3, P2_x, P2_y, A1, Epochs):
    difference = float('inf')
    h_loss = []
    h_epoch = []
    m = len(P0_y)  # very important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # pre_computing
    pre_hl_loss = HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, P0_y, B2, K2, P1_x, P1_y, B3, K3, P2_x, P2_y)
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
            bA_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - P0_y[i1] + KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 *
                        P1_x[i1] - B3 - K3 * P2_x[i1]) / m

            kA1_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - P0_y[i1] + (
                    KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 * P1_x[i1] - B3 - K3 * P2_x[i1])) * P0_x1[i1] / m

            kA2_grad += ((KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1) - P0_y[i1] + (
                    KA1 * P0_x1[i1] + KA2 * P0_x2[i1] + B1 - B2 - K2 * P1_x[i1] - B3 - K3 * P2_x[i1])) * P0_x2[i1] / m

            bAA_grad += ((K2 * P1_x[i1] + B2) - P1_y[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
                         B2 + K2 * P1_x[i1] + B3 + K3 * P2_x[i1]) / m

            kAA_grad += ((K2 * P1_x[i1] + B2) - P1_y[i1] - B1 - KA1 * P0_x1[i1] - KA2 * P0_x2[i1] +
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

        hl_loss = HL_LOSS(B1, KA1, KA2, P0_x1, P0_x2, P0_y, B2, K2, P1_x, P1_y, B3, K3, P2_x, P2_y)
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


# 5-fold-cross-validation
k = 5
num_validation_samples = len(p1_y) // k
p1_x = p1_x.reset_index(drop=True)
p1_y = p1_y.reset_index(drop=True)
p2_x = p2_x.reset_index(drop=True)
p2_y = p2_y.reset_index(drop=True)
p0_x = pd.concat([p1_x, p2_x], axis=1, ignore_index=True)
p0_y = p1_y + p2_y
MAE_p0_S = []
MAE_p1_S = []
MAE_p2_S = []
MAE_p0_h = []
MAE_p1_h = []
MAE_p2_h = []
MAE_p0_bu = []
MAE_p1_bu = []
MAE_p2_bu = []
MAE_p0_td = []
MAE_p1_td = []
MAE_p2_td = []
MCE_s = []
MCE_h = []
MCE_bu = []
MCE_td = []
for fold in range(k):
    print('This is {0} fold for cross validation'.format(fold + 1))
    testing_data_p1_x = p1_x[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_p1_y = p1_y[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_p1_x = pd.concat([p1_x[:num_validation_samples * fold], p1_x[num_validation_samples * (fold + 1):]])
    training_data_p1_y = pd.concat([p1_y[:num_validation_samples * fold], p1_y[num_validation_samples * (fold + 1):]])

    testing_data_p2_x = p2_x[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_p2_y = p2_y[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_p2_x = pd.concat([p2_x[:num_validation_samples * fold], p2_x[num_validation_samples * (fold + 1):]])
    training_data_p2_y = pd.concat([p2_y[:num_validation_samples * fold], p2_y[num_validation_samples * (fold + 1):]])

    testing_data_p0_x = p0_x[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    testing_data_p0_y = p0_y[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    training_data_p0_x = pd.concat([p0_x[:num_validation_samples * fold], p0_x[num_validation_samples * (fold + 1):]])
    training_data_p0_y = pd.concat([p0_y[:num_validation_samples * fold], p0_y[num_validation_samples * (fold + 1):]])

    testing_data_p1_x = testing_data_p1_x.reset_index(drop=True)
    testing_data_p1_y = testing_data_p1_y.reset_index(drop=True)
    training_data_p1_x = training_data_p1_x.reset_index(drop=True)
    training_data_p1_y = training_data_p1_y.reset_index(drop=True)

    testing_data_p2_x = testing_data_p2_x.reset_index(drop=True)
    testing_data_p2_y = testing_data_p2_y.reset_index(drop=True)
    training_data_p2_x = training_data_p2_x.reset_index(drop=True)
    training_data_p2_y = training_data_p2_y.reset_index(drop=True)

    testing_data_p0_x = testing_data_p0_x.reset_index(drop=True)
    testing_data_p0_y = testing_data_p0_y.reset_index(drop=True)
    training_data_p0_x = training_data_p0_x.reset_index(drop=True)
    training_data_p0_y = training_data_p0_y.reset_index(drop=True)
    a = 0.1
    # a = 0.000005
    epochs = 5000
    b1, ka1, ka2, b2, k2, b3, k3 = 0, 0, 0, 0, 0, 0, 0
    b1, ka1, ka2, b2, k2, b3, k3 = gd_for_HL(b1, ka1, ka2,
                                             training_data_p0_x[0],
                                             training_data_p0_x[1],
                                             training_data_p0_y, b2, k2,
                                             training_data_p1_x['years'],
                                             training_data_p1_y, b3, k3,
                                             training_data_p2_x['years'],
                                             training_data_p2_y, a, epochs)
    # print(b1, ka1, ka2, b2, k2, b3, k3)
    model_p1.fit(training_data_p1_x, training_data_p1_y)
    model_p2.fit(training_data_p2_x, training_data_p2_y)
    model_p0.fit(training_data_p0_x, training_data_p0_y)
    p1_y_test_pred = model_p1.predict(testing_data_p1_x)  # return list
    p2_y_test_pred = model_p2.predict(testing_data_p2_x)
    p0_y_test_pred = model_p0.predict(testing_data_p0_x)
    #     # Bottom Up
    p0_y_test_pred_bu = []
    for i in range(len(p1_y_test_pred)):
        p0_y_test_pred_bu.append(p1_y_test_pred[i] + p2_y_test_pred[i])
    print('MAE for BU model:(P1 P2 P0)')
    m1 = metrics.mean_absolute_error(testing_data_p1_y, p1_y_test_pred)
    m2 = metrics.mean_absolute_error(testing_data_p2_y, p2_y_test_pred)
    m3 = metrics.mean_absolute_error(testing_data_p0_y, p0_y_test_pred_bu)
    print(m1)
    print(m2)
    print(m3)
    MAE_p0_bu.append(m3)
    MAE_p2_bu.append(m2)
    MAE_p1_bu.append(m1)
    # top down
    p1_y_test_pred_td = []
    p2_y_test_pred_td = []
    proportion1, proportion2 = proportion(training_data_p0_y, training_data_p1_y, training_data_p2_y)

    for i in range(len(testing_data_p0_y)):
        p1_y_test_pred_td.append(proportion1 * p0_y_test_pred[i])
        p2_y_test_pred_td.append(proportion2 * p0_y_test_pred[i])
    print('MAE for TD model:(P1 P2 P0)')
    m1 = metrics.mean_absolute_error(testing_data_p1_y, p1_y_test_pred_td)
    m2 = metrics.mean_absolute_error(testing_data_p2_y, p2_y_test_pred_td)
    m3 = metrics.mean_absolute_error(testing_data_p0_y, p0_y_test_pred)
    print(m1)
    print(m2)
    print(m3)
    MAE_p0_td.append(m3)
    MAE_p2_td.append(m2)
    MAE_p1_td.append(m1)

    # base model
    print('MAE for Separate model:(P1 P2 P0)')
    m1 = metrics.mean_absolute_error(testing_data_p1_y, p1_y_test_pred)
    m2 = metrics.mean_absolute_error(testing_data_p2_y, p2_y_test_pred)
    m3 = metrics.mean_absolute_error(testing_data_p0_y, p0_y_test_pred)

    print(m1)
    print(m2)
    print(m3)
    MAE_p0_S.append(m3)
    MAE_p2_S.append(m2)
    MAE_p1_S.append(m1)

    # hierarchy model

    p0_y_predict = []
    p1_y_predict = []
    p2_y_predict = []
    for i in range(len(testing_data_p0_y)):
        p0_y_predict.append(ka1 * testing_data_p0_x[0][i] + ka2 * testing_data_p0_x[1][i] + b1)
        p1_y_predict.append(k2 * testing_data_p1_x['years'][i] + b2)
        p2_y_predict.append(k3 * testing_data_p2_x['years'][i] + b3)
    print('MAE for HRLS model:(P1 P2 P0)')
    m1 = metrics.mean_absolute_error(testing_data_p1_y, p1_y_predict)
    m2 = metrics.mean_absolute_error(testing_data_p2_y, p2_y_predict)
    m3 = metrics.mean_absolute_error(testing_data_p0_y, p0_y_predict)
    MAE_p0_h.append(m3)
    MAE_p2_h.append(m2)
    MAE_p1_h.append(m1)

    print(m1)
    print(m2)
    print(m3)

    # start to compute consistency error

    mce_h = 0
    for i in range(len(p1_y_predict)):
        mce_h += abs(p0_y_predict[i] - p1_y_predict[i] - p2_y_predict[i]) / len(p1_y_predict)
    MCE_h.append(mce_h)
    mce_s = 0
    for i in range(len(p0_y_test_pred)):
        mce_s += abs(p0_y_test_pred[i] - p1_y_test_pred[i] - p2_y_test_pred[i]) / len(p0_y_test_pred)
    MCE_s.append(mce_s)
    # 11111111111111111111111111111111111111111111111111111111111111111111111111111
    mce_bu = 0
    for i in range(len(p0_y_test_pred_bu)):
        mce_bu += abs(p0_y_test_pred_bu[i] - p1_y_test_pred[i] - p2_y_test_pred[i]) / len(p0_y_test_pred_bu)
    MCE_bu.append(mce_bu)

    mce_td = 0
    for i in range(len(p1_y_test_pred_td)):
        mce_td += abs(p0_y_test_pred[i] - p1_y_test_pred_td[i] - p2_y_test_pred_td[i]) / len(p1_y_test_pred_td)
    MCE_td.append(mce_td)

    # 11111111111111111111111111111111111111111111111111111111111111111111111111111
    print('For BU model mean consistence error CE={0}'.format(mce_bu))
    print('For TD model mean consistence error CE={0}'.format(mce_td))
    print('For Separate model mean consistence error CE={0}'.format(mce_s))
    print('For HRLS model consistence error CE={0}\n'.format(mce_h))

print('5-fold-cross-validation')
print('The average result for 5 fold cross validation : ')
print('MAE for separate model(P1,P2,P0)')
print(sum(MAE_p1_S) / k)
print(sum(MAE_p2_S) / k)
print(sum(MAE_p0_S) / k)
print('For Separate model mean consistence error MCE={0}'.format(sum(MCE_s) / k))
print('MAE for HRLS model:(P1,P2,P0)')
print(sum(MAE_p1_h) / k)
print(sum(MAE_p2_h) / k)
print(sum(MAE_p0_h) / k)
print('For HRLS model mean consistence error CE={0}'.format(sum(MCE_h) / k))
print('MAE for bottom-up model:(P1,P2,P0)')
print(sum(MAE_p1_bu) / k)
print(sum(MAE_p2_bu) / k)
print(sum(MAE_p0_bu) / k)
print('For BU model mean consistence error CE={0}'.format(sum(MCE_bu) / k))
print('MAE for TD model:(P1,P2,P0)')
print(sum(MAE_p1_td) / k)
print(sum(MAE_p2_td) / k)
print(sum(MAE_p0_td) / k)
print('For TD model consistence error CE={0}'.format(sum(MCE_td) / k))
