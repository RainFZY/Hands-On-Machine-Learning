import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import poisson
from scipy.special import expit

X = pd.read_csv('data/Bayes_classifier/X.csv', header=None)  # 4600 * 54
y = pd.read_csv('data/Bayes_classifier/y.csv', header=None, names=['pred'])  # 4600 * 1

data = pd.concat([X, y], axis=1)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

"""""""""""""""""""""""""""""""""""""""""""""
                Problem 2.1
"""""""""""""""""""""""""""""""""""""""""""""


def naive_bayes_classifier(pai, x0, lamb):
    D = len(x0)
    ans = [1 - pai, pai]
    # naive Bayes classifier formula from the question
    for y in range(2):
        arg = [poisson.pmf(x0[d], lamb[y][d]) for d in range(D)]  # 1 * 54 list
        ans[y] *= np.prod(arg)

    return np.argmax(ans)


def implement_naive_bayes_classifier(X, y, fold_number=10):
    D = X.shape[1]  # 54
    fold = KFold(n_splits=fold_number, shuffle=True)  # randomly partition the data into 10 groups
    prediction_table = []
    lamb_all = []

    # 10-fold, 10 runs
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        len_train = X_train.shape[0]  # train size: 4140
        # Get pai using Problem 1 (a)'s answer
        pai = y_train.mean()

        # Get λ using Problem 1 (b)'s answer
        lamb = np.zeros((2, D))  # λ(y,d)，y 2 dimensions, d 54 dimensions
        for flag in range(2):
            for d in range(D):
                numerator = sum(X_train.iloc[i][d] * (y_train.iloc[i] == flag) for i in range(len_train))
                denominator = sum(y_train.iloc[i] == flag for i in range(len_train))
                lamb[flag][d] = numerator / denominator

        y_pred = np.zeros(len(X_test))  # 460 * 1
        for i in range(len(X_test)):
            y_pred[i] = naive_bayes_classifier(pai, X_test.iloc[i], lamb)

        temp_table = np.zeros((2, 2))
        for m in [0, 1]:
            for n in [0, 1]:
                temp_table[m][n] = sum([(y_test.values[i] == m) & (y_pred[i] == n) for i in range(len(y_pred))])

        prediction_table.append(temp_table)
        lamb_all.append(lamb)

    prediction_table = sum(prediction_table) / 10  # average
    lamb_all = sum(lamb_all) / 10  # average

    return lamb_all, prediction_table

# temp_table: 2 * 2 matrix, lamb: 2 * 54
lamb, temp_table = implement_naive_bayes_classifier(X, y)
lamb_hat_mean = np.array(lamb).mean(axis=0)  # 1 * 54
accu = (temp_table[0][0] + temp_table[1][1]) / temp_table.sum()

print(temp_table)
print("Accuracy of Naive Bayes Classifier is %.3f" % accu)


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 2.2
"""""""""""""""""""""""""""""""""""""""""""""


def stem_plot(lamb):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), dpi=300)

    ax1.grid()
    ax1.stem(lamb[0], markerfmt='ro', basefmt='black')
    ax1.set_title('54 average Poisson parameters λ0,d (y=0)')
    ax1.set_xlabel('d (54 dimensions)')
    ax1.set_ylabel('λ0,d (y=0)')

    ax2.grid()
    ax2.stem(lamb[1], markerfmt='ro', basefmt='black')
    ax2.set_title('54 average Poisson parameters λ1,d (y=1)')
    ax2.set_xlabel('d (54 dimensions)')
    ax2.set_ylabel('λ1,d (y=1)')

    plt.savefig('2.2_Poisson Parameters_λ.png')
    plt.show()


stem_plot(lamb)


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 2.3
"""""""""""""""""""""""""""""""""""""""""""""


def logistic_regression(X, y, fold_number=10, iteration=1000):
    X['x0'] = 1  # add a dimension equal to +1 to each data point
    y[y == 0] = -1  # set y=-1 if y==0

    # data preparation
    D = X.shape[1]  # 55
    fold = KFold(n_splits=fold_number)
    eta = 0.01 / 4600

    # record 10 output
    objection_function_list = []

    # 10 runs
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        len_train = X_train.shape[0]  # 4140
        w = np.zeros(D)  # 1 * 55
        loss_function = []
        # 1001 runs
        for ite in range(iteration + 1):
            derivative = np.zeros(D)  # 1 * 55
            for i in range(len_train):
                # expit(): sigmoid function, expit(x)= 1 / (1 + exp(-x))
                # σ(yw) in slides = expit(yxw) in python
                temp = expit(y_train.values[i] * X_train.values[i].dot(w))
                derivative += (1 - temp) * y_train.values[i] * X_train.values[i]
            # log(): ln()
            loss_function.append(
                sum(np.log(expit(y_train.values[i] * X_train.values[i].dot(w))) for i in range(len_train)))
            w += eta * derivative  # w: 1 * 55
        objection_function_list.append(loss_function)

    return objection_function_list


# objection_function_list: 10 * 1 * 1001
objection_function_list = logistic_regression(X, y)


def plot_objective_func(loss_func_list):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.grid()
    for i in range(10):
        plt.plot(loss_func_list[i])
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.title('Logistic Regression Objective Training Function L, 10 Groups')
    plt.savefig('2.3_Logistic Regression Objective Training Function.png')
    plt.show()


plot_objective_func(objection_function_list)


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 2.4
"""""""""""""""""""""""""""""""""""""""""""""


def newton_method(X, y, fold_number=10, iteration=100):
    X['x0'] = 1  # add a dimension equal to +1 to each data point
    y[y == 0] = -1  # set y=-1 if y==0

    # data preparation
    D = X.shape[1]  # 55
    fold = KFold(n_splits=fold_number)
    objection_function_list = []
    prediction_table = []
    # 1001 runs
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        len_train = X_train.shape[0]  # 4140
        w = np.zeros(D)  # 1 * 55
        loss_function = []
        for ite in range(iteration + 1):
            derivative = np.zeros(D)
            for i in range(len_train):
                temp = expit(y_train.values[i] * X_train.values[i].dot(w))  # float number
                derivative += (1 - temp) * y_train.values[i] * X_train.values[i]  # X_train.values[i]: 1*55

            second_derivative = np.zeros((D, D))
            for i in range(len_train):
                temp = expit(y_train.values[i] * X_train.values[i].dot(w))  # float number
                second_derivative -= float(temp * (1 - temp)) * X_train.values[i].T * X_train.values[i]
            second_derivative_inv = np.linalg.inv(second_derivative + np.diag(([1e-6] * D)))

            loss_function.append(
                sum(np.log(expit(y_train.values[i] * X_train.values[i].dot(w))) for i in range(len_train)))
            w -= np.dot(second_derivative_inv, derivative)  # update w

        objection_function_list.append(loss_function)

        # predict y
        y_pred = expit(X_test.values.dot(w))
        y_pred[y_pred < 0.5] = -1
        y_pred[y_pred >= 0.5] = 1

        temp_table = []
        for m in [-1, 1]:
            for n in [-1, 1]:
                temp_table.append(sum([(y_test.values[i] == m) & (y_pred[i] == n) for i in range(len(y_pred))]))
        temp_table = np.array(temp_table).reshape(2, 2)
        prediction_table.append(temp_table)

    prediction_table = sum(prediction_table)

    return objection_function_list, prediction_table


def plot_newton_objective_function(loss_func_list):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.grid()
    for i in range(10):
        plt.plot(loss_func_list[i])
    plt.xlabel('Iterations')
    plt.ylabel('Newton Objective Function')
    plt.title('Logistic Regression Objective Function L Using Newton Method, 10 Groups')
    plt.savefig('2.4_Newton Method Objective Function2.png')
    plt.show()


objection_function_list, cross_table = newton_method(X, y)
plot_newton_objective_function(objection_function_list)
print(cross_table)
acc = (cross_table[0][0] + cross_table[1][1]) / cross_table.sum()
print('testing accuracy: %.3f'%acc)


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 3.1
"""""""""""""""""""""""""""""""""""""""""""""


data_dir = 'data/Gaussian_process/'
X_train = pd.read_csv(data_dir + 'X_train.csv')
X_test = pd.read_csv(data_dir + 'X_test.csv')
y_train = pd.read_csv(data_dir + 'y_train.csv', names=['y'])
y_test = pd.read_csv(data_dir + 'y_test.csv', names=['y'])


# Gaussian Kernel indicated in the question
def kernel(x_i, x_j, b):
    # return a num
    return np.exp(-sum((x_i - x_j)**2) / b)


def gaussian_process_model(b, sigma_square, X_train, y_train, X_test):
    n = X_train.shape[0]  # 350
    p = X_test.shape[0]  # 42

    # Kij: 350 * 350
    # X_train[i], X_train[j]: ndarray(7,1)
    Kij = np.array([[kernel(X_train[i], X_train[j], b) for i in range(n)] for j in range(n)])
    inv = np.linalg.inv(np.diag([sigma_square] * n) + Kij)

    miu = np.zeros(p)  # 1 * 42
    Sigma = np.zeros(p)  # 1 * 42

    # 42 runs, iterate test set and compute μ and Σ
    for j in range(p):  # for every new point x0 in testing data.
        x0 = X_test[j]
        K_Dn = np.zeros(n)  # 350 * 1
        for i in range(n):
            # X_train[i], x0: ndarray(7,1)
            K_Dn[i] = kernel(X_train[i], x0, b)  # calculate every item in K_Dn

        # use formula from slides24 to calculate new distribution parameters
        miu[j] = K_Dn.dot(inv).dot(y_train)[0]
        Sigma[j] = sigma_square + kernel(x0, x0, b) - K_Dn.dot(inv).dot(K_Dn.T)

    # 1*42, 1*42
    return miu, Sigma


RMSE = []
for b in [5, 7, 9, 11, 13, 15]:
    for sigma_square in np.arange(0.1, 1.1, .1):
        miu, Sigma = gaussian_process_model(b, sigma_square, X_train.values, y_train.values, X_test.values)
        RMSE.append(np.sqrt(sum((y_test.values.T[0] - miu) ** 2) / X_test.values.shape[0]))

RMSE = np.array(RMSE).reshape(6, 10)
print(RMSE)


"""""""""""""""""""""""""""""""""""""""""""""
                Problem 3.3
"""""""""""""""""""""""""""""""""""""""""""""

# choose the 4th dimension
X_train4 = X_train[3].values.reshape(-1, 1)  # ndarray(350, 1)
b = 5
sigma_square = 2.0
# First Part
fig = plt.figure(dpi=300)
plt.grid()
# a scatter plot of the data (x[4] versus y for each point)
plt.scatter(X_train4, y_train, c='red', label='training data')

# Second Part
_miu, _Sigma = gaussian_process_model(b, sigma_square, X_train4, y_train.values, X_train4)  # ndarray(350,), ndarray(350,)
mean = np.array([X_train4[:, 0], _miu]).T  # ndarray(350, 2)
mean = mean[mean[:, 0].argsort()].T  # ndarray(2, 350)
plt.plot(mean[0], mean[1], color='blue', label='predictive mean')
plt.xlabel('x[4](car weight)')
plt.ylabel('y')
plt.title('x[4] versus y & predictive mean of the Gaussian process')
plt.legend()
plt.savefig('3.3_x[4] versus y_and_predictive mean of the Gaussian process4.png')
plt.show()