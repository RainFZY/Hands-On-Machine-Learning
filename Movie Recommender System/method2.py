import numpy as np
import matplotlib.pyplot as plt

critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snake on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},

           'Gene Seymour': {'Lady in the Water': 3.0, 'Snake on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5, },

           'Michael Phillips': {'Lady in the Water': 2.5, 'Snake on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},

           'Claudia Puig': {'Snake on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},

           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snake on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},

           'Jack Matthews': {'Lady in the Water': 3.0, 'Snake on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},

           'Toby': {'Snake on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}


def matrix(R, P, Q, K, alpha, beta):
    result = []
    steps = 1
    while 1:
        # 使用梯度下降的一步步的更新P,Q矩阵直至得到最终收敛值
        steps = steps + 1
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # .dot(P,Q) 表示矩阵内积,即Pik和Qkj k由1到k的和eij为真实值和预测值的之间的误差,
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    # 求误差函数值，我们在下面更新p和q矩阵的时候我们使用的是化简得到的最简式，较为简便，
                    # 但下面我们仍久求误差函数值这里e求的是每次迭代的误差函数值，用于绘制误差函数变化图
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        # 在上面的误差函数中加入正则化项防止过拟合
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))

                    for k in range(K):
                        # 在更新p,q时我们使用化简得到了最简公式
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        print('迭代轮次:', steps, '   e:', e)
        result.append(e)  # 将每一轮更新的损失函数值添加到数组result末尾
        # 当损失函数小于一定值时，迭代结束
        if eij < 0.00001:
            break
    return P, Q, result


R = [
    [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
    [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
    [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
    [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
    [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
    [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
    [0.0, 4.5, 0.0, 4.0, 1.0, 0.0]
]

R = np.array(R)

alpha = 0.0001  # 学习率
beta = 0.002  #

N = len(R)
M = len(R[0])
K = 2

p = np.random.rand(N, K)  # 随机生成一个 N行 K列的矩阵
q = np.random.rand(K, M)  # 随机生成一个 M行 K列的矩阵

P, Q, result = matrix(R, p, q, K, alpha, beta)
print("矩阵Q为：\n", Q)
print("矩阵P为：\n", P)
print("矩阵R为：\n", R)
MF = np.dot(P, Q)
print("预测矩阵：\n", MF)

# 下面代码可以绘制损失函数的收敛曲线图
n = len(result)
x = range(n)
print(x)
plt.plot(x, result, color='b', linewidth=3)
plt.xlabel("generation")
plt.ylabel("loss")
plt.show()

