import numpy as np
import matplotlib.pyplot as plt


def ab_clusters():
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean1, cov1, 300)

    mean2 = [1, 2]
    cov2 = [[1, 0], [0, 2]]
    data = np.append(data,
                     np.random.multivariate_normal(mean2, cov2, 200),
                     0)
    print(np.round(data, 4))
    return np.round(data, 4)


def a_clusters():
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean1, cov1, 300)

    return np.round(data, 4)  # 四舍五入后保留四位小数

def b_clusters():
    mean2 = [1, 2]
    cov2 = [[1, 0], [0, 2]]
    data = np.random.multivariate_normal(mean2, cov2, 200)

    return np.round(data, 4)  # 四舍五入后保留四位小数


def save_data(data, filename):
    with open(filename, 'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i, 0]) + ',' + str(data[i, 1]) + '\n')


def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data.append([float(i) for i in line.split(',')])
    return np.array(data)


def show_a_scatter(data):
    x, y = data.T
    plt.scatter(x, y)
    plt.axis()
    plt.title("a_scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def show_b_scatter(data):
    x, y = data.T
    plt.scatter(x, y)
    plt.axis()
    plt.title("b_scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def show_ab_scatter(data):
    x, y = data.T
    plt.scatter(x, y)
    plt.axis()
    plt.title("ab_scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


data_a = a_clusters()
data_b = b_clusters()
data_ab = ab_clusters()
save_data(data_a, 'a_clusters.txt')
save_data(data_b, 'b_clusters.txt')
save_data(data_ab, 'ab_clusters.txt')
d1 = load_data('a_clusters.txt')
d2 = load_data('b_clusters.txt')
d3 = load_data('ab_clusters.txt')
show_a_scatter(d1)
show_b_scatter(d2)
show_ab_scatter(d3)
