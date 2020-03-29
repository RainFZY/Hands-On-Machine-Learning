from numpy import *
import operator
import numpy as np
import matplotlib.pyplot as plt
# encoding=utf-8
from matplotlib.colors import ListedColormap
from pylab import *
import math
from sklearn import neighbors
from sklearn.datasets.samples_generator import make_classification


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False #不然坐标下的编号会出现乱码


# 从.txt文件中读取数据
def loadData():

    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean1, cov1, 300)

    mean2 = [1, 2]
    cov2 = [[1, 0], [0, 2]]
    data = np.append(data,
                     np.random.multivariate_normal(mean2, cov2, 200),
                     0)
    # print(np.round(data, 4))

    labels = []
    for i in range(300):
        labels.append(0) # 0代表A点

    for i in range(200):
        labels.append(1) #　1代表Ｂ点

    data = np.round(data, 4)# 保留四位小数

    return data,labels


dataSet,labels = loadData() # 特征值矩阵，标签矩阵
#print(dataSet)
#print(labels)


# 绘制生成的ab点分布图
x1, y1 = dataSet[0:300].T # 所有A类点
x2, y2 = dataSet[300:500].T # 所有B类点
plt.scatter(x1, y1, s = 20, c = 'r', marker = '.')
plt.scatter(x2, y2, s = 20, c = 'b', marker = 'x')
plt.axis()
plt.title("ab_distribution")
plt.xlabel("x")
plt.ylabel("y")

a = np.ones(500) # 500个1的矩阵
dataMat = np.insert(dataSet, 0, values=a, axis=1) # 在特征值矩阵第0列加一列1，之后是为了跟常数项相乘

# 定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 定义梯度上升算法函数，其中dataMatIn参数为2维numpy数组，每列代表不同的特征，每行代表一个训练样本，labelMat为标签分类
def gradAscent(dataMatIn, classLabels, epoch):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix，dataMatrix是500 x 3矩阵
    #print(dataMatrix.transpose())
    #print(classLabels)
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix， 500 x 1
    #print(labelMat)
    m, n = shape(dataMatrix)  # 获取矩阵的行和列，m = 500，n = 3
    alpha = 0.001 # α为步长，也就是学习速率，控制更新的幅度
    # epoch = 500 # 迭代次数，也就是epoch
    theta = ones((n, 1)) # 3 x 1矩阵,每一行是一个1
    # print(dataMatrix)

    # 迭代
    for k in range(epoch):
        # dataMatrix矩阵每一行的三个数是1,x,y；theta矩阵一列三个数依次是常数项、x、y的系数
        # 两个矩阵相乘后变成500*1矩阵，每一行的值就是wei[0] + wei[1] * dataSet[i][0] + wei[2] * dataSet[i][1]
        # 即拟合直线，大于0小于0来判断属于A还是属于B，作为横坐标值代入sigmoid函数正好合适

        # 计算预测值域实际值的偏差，500 x 1
        h = sigmoid(dataMatrix * theta)  # 括号内大于0，h更接近于1，B；括号内小于0，h更接近于0，A。h共500行，每一行代表一个点
        error = (labelMat - h)

        # transpose转置，dataMatrix.transpose()是3 x 500
        # matrix mult，梯度下降算法，找出最佳的参数，theta是3*1矩阵
        theta = theta + alpha * dataMatrix.transpose() * error

    # theta就是参数列向量(要求解的),表示回归系数[w0,w1,w2]
    return theta



# 画出决策边界：画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(wei):
    # 导入数据
    dataArr = array(dataSet)  # dataMat转换为数组，dataSet是全局变量
    #print(dataArr)
    n = shape(dataArr)[0] # n=500
    xcord1 = [] # A点样本
    ycord1 = []
    xcord2 = [] # B点样本
    ycord2 = []

    # 将数据按类别分类
    for i in range(n):
        if int(labels[i]) == 1:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
        else:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='r', marker='.') # s是点的大小
    ax.scatter(xcord2, ycord2, s=20, c = 'b', marker = 'x')
    x = arange(-3.0, 4.0, 0.1) # 绘制的线的显示范围，最小x，最大x
    y = (-wei[0] - wei[1] * x) / wei[2] # 绘制的函数方程
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


best_accuracy = 0
best_epoch = 0
epoch_list = []
accuracy_list = []
for epoch in range(1, 152, 5):
    theta = gradAscent(dataMat, labels, epoch)
    #print('theta:\n',theta)
    #print(theta.getA())

    # 计算准确率
    wei = theta.getA()
    correctNum = 0
    for i in range(500):
        if(i < 300):
            if(wei[0] + wei[1] * dataSet[i][0] + wei[2] * dataSet[i][1] < 0):
                correctNum += 1
        else:
            if (wei[0] + wei[1] * dataSet[i][0] + wei[2] * dataSet[i][1] > 0):
                correctNum += 1

    accuracy = correctNum/500
    if(accuracy > best_accuracy):
        best_accuracy = accuracy
        best_epoch = epoch
    print('epoch = ',epoch, 'accuracy = ',accuracy)
    epoch_list.append(epoch)
    accuracy_list.append(accuracy)

print('min-best epoch = ', best_epoch, ', best accuracy = ', best_accuracy)

# 绘制epoch-accuracy曲线
plt.figure()
names = epoch_list
x = range(len(names))
y = accuracy_list
plt.plot(x, y, marker='o', mec='r', mfc='w')
plt.xticks(x, names)
plt.xlabel(u"epoch") #X轴标签
plt.ylabel("accuracy") #Y轴标签
plt.title("不同epoch下的accuracy 曲线图")

# 用best epoch来迭代计算theta
best_theta = gradAscent(dataMat, labels, best_epoch)
# 调用函数绘图
plotBestFit(best_theta.getA()) # #将矩阵转换为数组，返回权重数组


