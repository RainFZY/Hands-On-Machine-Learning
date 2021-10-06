import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

label_dic = {
    0: 'cylinders',
    1: 'displacement',
    2: 'horsepower',
    3: 'weight',
    4: 'acceleration',
    5: 'year made',
    6: 'constant'
}
data_dir = 'hw1-data/'
X_train = pd.read_csv(data_dir + 'X_train.csv', header=None, names=label_dic.values())
X_test = pd.read_csv(data_dir + 'X_test.csv', header=None, names=label_dic.values())
y_train = pd.read_csv(data_dir + 'y_train.csv', header=None)
y_test = pd.read_csv(data_dir + 'y_test.csv', header=None)

def rmse(y_t, y_p):
    sum_delta = sum((y_t[c] - y_p[c]) ** 2 for c in range(42))
    sum_delta = np.sqrt(sum_delta / 42.0)
    return sum_delta

def construct_polynomial_matrix(data, p):
    """Preprocesses dataframe for p-polynomial regression.
    According to the instruction, added p-polynomial dimension requires standardization

    Args:
        data: input original dataframe.
        p: p-polynomial process.

    Returns:
        A dataframe with more dimensions which derive from polynomial process.
    """
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


def get_RMSE(X_train, X_test, y_train, y_test, max_lambda):
    """Gets a list of RMSE versus lambda in p-polynomial condition.
    Args:
        X_train, ... , y_test: training and testing data.
        max_lambda: the range of lambda.
    Returns:
        RMSE lists.
    """
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
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.title('RMSE versus lambda curves')
# p-th polynomial process
for p in range(1, 4):
    # preprocess X data
    # ----- standardize train, test together -----
    X_p = pd.concat([X_train, X_test], axis=0)
    X_p = construct_polynomial_matrix(X_p, p)
    X_train_p = X_p.iloc[:len(X_train)]
    X_test_p = X_p.iloc[-len(X_test):]
    # ----- standardize train, test separately -----
    # X_train_p = construct_polynomial_matrix(X_train, p)
    # X_test_p = construct_polynomial_matrix(X_test, p)

    y_list = get_RMSE(X_train_p, X_test_p, y_train, y_test, 101)
    plt.plot(y_list, label='Polynomial regression of degree p=%d' % p)
plt.legend()
# plt.savefig('RMSE versus lambda curves in polynomial regression.png')
plt.show()