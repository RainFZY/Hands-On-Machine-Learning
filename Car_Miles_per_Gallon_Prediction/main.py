import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year made', 'constant']

# Read csv files
X_train = pd.read_csv('hw1-data/X_train.csv', header=None, names=features)
X_test = pd.read_csv('hw1-data/X_test.csv', header=None, names=features)
y_train = pd.read_csv('hw1-data/y_train.csv', header=None)
y_test = pd.read_csv('hw1-data/y_test.csv', header=None)
# print(X_test)

n = X_train.shape[1]  # 7

"""""""""""""""""""""""""""""""""""""""""""""
                Problem 3.1
"""""""""""""""""""""""""""""""""""""""""""""
# Calculate WRR
# formula: WRR = inverse(λI + XTX) XT y
X_train, y_train = np.array(X_train.values), np.array(y_train.values)
WRR = []
for Lambda in range(5001):
    temp = np.diag(np.ones(n) * Lambda) + X_train.T.dot(X_train)
    temp = np.linalg.inv(temp)
    temp = temp.dot(X_train.T)
    temp = temp.dot(y_train)
    temp = temp.T[0]  # shape: 1 * 7
    WRR.append(temp)

WRR = np.array(WRR)  # shape: 5001 * 7

# SVD
U, S, vT = np.linalg.svd(X_train)

# df(λ): Degrees of freedom, as x-coordinate
df = np.zeros(5001)
for lamb in range(5001):
    for i in range(n):
        df[lamb] += (S[i] ** 2) / (S[i] ** 2 + lamb)
# print(df)

# plot
fig = plt.figure()
plt.grid()
plt.xlabel('df(λ)')
plt.ylabel('WRR', rotation='horizontal', loc='center')
plt.title('WRR - df(λ) curves')
colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'black']
# values of df as x-coordinate, not λ
for i in range(n):
    plt.plot(df, WRR.T[i], color=colors[i], label=features[i])
plt.legend()  # add legend for plot
plt.savefig('WRR-df(λ) curves.png')
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""
                Problem 3.3
"""""""""""""""""""""""""""""""""""""""""""""
WRR_test = WRR[:51]  # 51 * 7
X_test, y_test = np.array(X_test.values), np.array(y_test.values)
y_pred = (X_test.dot(WRR_test.T)).T  # 51 * 42

def rmse(y_test, y_pred):
    res = 0
    for i in range(42):
        res += sum((y_test[i] - y_pred[i]) ** 2)
    res = np.sqrt(res / 42.0)
    return res

y_list = np.zeros(51)
for i in range(51):
    y_list[i] = rmse(y_test, y_pred[i])

fig = plt.figure()
plt.grid()
plt.xlabel('λ')
plt.ylabel('RMSE', rotation='horizontal', loc='center')
plt.title('RMSE - λ')
plt.plot(y_list, color='red')
plt.savefig('RMSE-λ curves.png')
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 3.4
"""""""""""""""""""""""""""""""""""""""""""""
X_train = pd.read_csv('hw1-data/X_train.csv', header=None, names=features)
X_test = pd.read_csv('hw1-data/X_test.csv', header=None, names=features)
y_train = pd.read_csv('hw1-data/y_train.csv', header=None)
y_test = pd.read_csv('hw1-data/y_test.csv', header=None)

def construct_polynomial_matrix(data, p):
    data = data[data.columns[:-1]]  # delete the last all-1 column
    dt = [data]

    # get new data with p-polynomial process
    for i in range(2, p + 1):
        data_i = data ** i
        # standardization, 分母是std还是range?
        for column in data_i.columns:
            data_i[column] = (data_i[column] - data_i[column].mean()) / data_i[column].std()
        # column name: w --> w^2
        data_i = data_i.rename(columns=lambda x: x + '^' + str(i))
        dt.append(data_i)  # [[w1,w2]] --> [[w1,w2],[w1^2,w2^2]]
    dt = pd.concat(dt, axis=1)  # dt: list --> DataFrame, w1,w2,w1^2,w2^2
    dt['constant'] = 1  # w1,w2,w1^2,w2^2 --> w1,w2,w1^2,w2^2,constant

    return dt


def calculate_RMSE(X_train, X_test, y_train, y_test, max_lambda):
    n = X_train.shape[1]  # get the dimension of new training data
    WRR = []
    X, y = np.array(X_train.values), np.array(y_train.values)

    # calculate WRR for each possible lambda
    for Lambda in range(max_lambda):
        temp = np.diag(np.ones(n) * Lambda) + X.T.dot(X)
        temp = np.linalg.inv(temp)
        temp = temp.dot(X.T)
        temp = temp.dot(y)
        temp = temp.T[0]  # shape: 1 * 7
        WRR.append(temp)
    WRR = np.array(WRR)
    y_pred = X_test.dot(WRR.T)  # get prediction of y, which is a function of lambda

    res = np.zeros(max_lambda)
    for Lambda in range(max_lambda):
        res[Lambda] = rmse(y_test.values, y_pred[Lambda].values)
    # return a list of RMSE values
    return res


fig = plt.figure()
plt.grid()
plt.xlabel('λ')
plt.ylabel('RMSE', rotation='horizontal', loc='center')
plt.title('pth-order polynomial regression model: RMSE - λ curves')
colors = ['purple', 'orange', 'cyan']
# p-th polynomial process
for p in range(1, 4):
    # construct polynomial matrix
    # ----- standardize train, test together -----
    X_p = pd.concat([X_train, X_test], axis=0)
    X_p = construct_polynomial_matrix(X_p, p)
    X_train_p = X_p[:len(X_train)]
    X_test_p = X_p[-len(X_test):]
    # ----- standardize train, test separately -----
    # X_train_p = construct_polynomial_matrix(X_train, p)
    # X_test_p = construct_polynomial_matrix(X_test, p)

    y_list = calculate_RMSE(X_train_p, X_test_p, y_train, y_test, 101)
    plt.plot(y_list, color=colors[p-1], label='p = %d' % p)
plt.legend()
plt.savefig('pth-order RMSE-λ curves.png')
plt.show()

