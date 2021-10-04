import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year made', 'w0']

# Read csv files
data_dir = 'hw1-data/'
X_train = pd.read_csv(data_dir + 'X_train.csv', header=None, names=features)
X_test = pd.read_csv(data_dir + 'X_test.csv', header=None, names=features)
y_train = pd.read_csv(data_dir + 'y_train.csv', header=None)
y_test = pd.read_csv(data_dir + 'y_test.csv', header=None)
# print(X_test)

dim = X_train.shape[1]  # dimension
WRR = []
X = X_train.values

# Calculate WRR
# formula: WRR = inverse(λI + XTX) XT y
for lamb in range(5001):
    lambI = np.diag(np.ones(dim) * lamb)
    inv = np.linalg.inv(lambI + X.T.dot(X))
    WRR.append(inv.dot(X.T).dot(y_train.values).T[0])
WRR = np.array(WRR)

# SVD
U, S, vT = np.linalg.svd(X)

# df(λ): Degrees of freedom
df = np.zeros(5001)
for lamb in range(5001):
    for i in range(7):
        df[lamb] += (S[i] ** 2) / (S[i] ** 2 + lamb)
print(df)

# plot
fig = plt.figure(figsize = (12, 8), dpi=300)
plt.grid()
plt.xlabel('df(λ)')
plt.ylabel('wRR')
plt.title('wRR versus df(λ) curves')
for i in range(7):
    plt.plot(df, WRR.T[i], label=features[i])
plt.legend()  # add legend for plot
plt.savefig('WRR versus df(λ) curves.png')
plt.show()



