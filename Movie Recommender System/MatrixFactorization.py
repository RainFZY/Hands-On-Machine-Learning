from math import *
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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


def SGD(R,P,Q,K,epoch=5000,alpha=0.002,beta=0.02): # 矩阵分解，epoch：梯度下降次数；alpha：步长；beta：β。
    Q = Q.T # 矩阵的转置
    loss = []
    # 梯度下降
    for step in range(epoch):
        e = 0
        for i in range(len(R)): # 遍历行
            for j in range(len(R[i])): # 遍历列
                eij = R[i][j] - np.dot(P[i,:],Q[:,j]) # 求残差值，.dot表示矩阵相乘
                for k in range(K):
                    if R[i][j]>0:        #限制评分大于零
                        P[i][k] = P[i][k] + alpha*(2*eij*Q[k][j]-beta*P[i][k]) # 加入正则化，更新P
                        Q[k][j] = Q[k][j] + alpha*(2*eij*P[i][k]-beta*Q[k][j]) # 加入正则化，更新Q
                # 计算损失值
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)  # 损失值的和
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))  # 加入正则化后的损失值的和
        loss.append(e)
        if e<0.001: # 收敛条件，0.001为阈值
            break
    return P,Q.T,loss

if __name__ == '__main__':   #主函数
    R = [
        [2.5, 3.5, 3  , 3.5, 2.5, 3  ],
        [3  , 3.5, 1.5, 5  , 3.5, 3  ],
        [2.5, 3  , 0  , 3.5, 0  , 4  ],
        [0  , 3.5, 3  , 4  , 2.5, 4.5],
        [3  , 4  , 2  , 3  , 2  , 3  ],
        [3  , 4  , 0  , 5  , 3.5, 3  ],
        [0  , 4.5, 0  , 4  , 1  , 0  ]
    ]
    print("原矩阵R：\n", R)
    markTable = PrettyTable(
        ['Lady in the Water', 'Snake on a Plane', 'Just My Luck', 'Superman Returns', 'You, Me and Dupree','The Night Listener'])
    for i in range(7):
        markTable.add_row(R[i])
    markTable.add_column('Movies', ['Lisa Rose', 'Gene Seymour', 'Michael Phillips', 'Claudia Puig', 'Mick LaSalle','Jack Matthews', 'Toby'])
    print("用户评分表：\n", markTable)

    R=np.array(R)
    N=len(R)    #原矩阵R的行数
    M=len(R[0]) #原矩阵R的列数
    K=2    #K值可根据需求改变
    P=np.random.rand(N,K) #随机生成一个 N行 K列的矩阵
    Q=np.random.rand(M,K) #随机生成一个 M行 K列的矩阵
    nP,nQ,loss = SGD(R,P,Q,K)

    print("矩阵Q：\n", Q)
    print("矩阵P：\n", P)

    R_MF = np.dot(nP,nQ.T)
    print("推荐矩阵：\n",R_MF)

    recTable = PrettyTable(['Lady in the Water','Snake on a Plane','Just My Luck',
                            'Superman Returns','You, Me and Dupree','The Night Listener'])
    for i in range(7):
        recTable.add_row(R_MF[i])
    recTable.add_column('Movies',['Lisa Rose','Gene Seymour','Michael Phillips',
                                  'Claudia Puig','Mick LaSalle','Jack Matthews','Toby'])
    print("推荐指数表：\n",recTable)

    recToby = PrettyTable(['Lady in the Water','Just My Luck','The Night Listener'])
    recToby.add_row([R_MF[6][0],R_MF[6][2],R_MF[6][5]])
    print("给Toby的电影推荐指数：\n",recToby)

    # 画图
    plt.plot(range(len(loss)),loss)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
