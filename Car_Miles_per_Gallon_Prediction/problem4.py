import pandas as pd
import numpy as np
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

# data = pd.array([[2,1], [3,1], [4,1]])
data1 = {
    "w1":[1,2,3],
    "w2":[4,5,6],
    "w3":[1,1,1]
}
df1 = pd.DataFrame(data1)
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

X_p = pd.concat([X_train, X_test], axis=0)  # axis=0跨行

X_p = construct_polynomial_matrix(X_p, 3)
X_train_p = X_p[:len(X_train)]
X_test_p = X_p.iloc[-len(X_test):]


X_train_p1 = construct_polynomial_matrix(X_train, 3)
print(X_train_p == X_train_p1)

dim_p = X_train_p.shape[1]