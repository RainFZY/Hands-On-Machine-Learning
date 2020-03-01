from numpy import *
import operator
import numpy as np
import matplotlib.pyplot as plt
# encoding=utf-8
from matplotlib.colors import ListedColormap
from pylab import *
from sklearn import neighbors
from sklearn.datasets.samples_generator import make_classification
mpl.rcParams['font.sans-serif'] = ['SimHei']



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
        labels.append('1') # 1代表A点

    for i in range(200):
        labels.append('2') #　２代表Ｂ点

    # 保留四位小数
    return np.round(data, 4),labels


#处理数据
#计算已知类别数据集中的点与当前点之间的距离（欧式距离）
#按照距离递增次序排序
#选取与当前点距离最小的K个点
#确定前K个点所在类别的出现频率
#返回前k个点出现频率最高的类别最为当前点的预测分类
#inX输入向量，训练集dataSet,标签向量labels，k表示用于选择最近邻的数目
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #读取dataSet的行数
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #把inX展开成dataSetSize行，每行都一样的数组，跟dataSet做差分
    sqDiffMat = diffMat ** 2 #平方
    sqDistances = sqDiffMat.sum(axis=1) #把每一行求和
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort() #将数组的值从小到大排序后，并按照其相对应的索引值输出.
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
		key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


dataSet,labels = loadData()
# print(labels)

'''
trainingSet = np.vstack((dataSet[0:270],dataSet[300:480]))
testSet = np.vstack((dataSet[270:300], dataSet[480:500]))

trainingLabels = labels[0:270] + labels[300:480]
testLabels = labels[270:300] + labels[480:500]
'''

# 绘制生成的ab点分布图
x1, y1 = dataSet[0:300].T # 所有A类点
x2, y2 = dataSet[300:500].T # 所有B类点
plt.scatter(x1, y1, c = 'k', marker = '.')
plt.scatter(x2, y2, c = 'b', marker = 'x')
plt.axis()
plt.title("ab_distribution")
plt.xlabel("x")
plt.ylabel("y")


# 测试正确率，选出最佳k值
outputLabels = []
correctCount = 0
correctCountSum = 0
best_accuracy = 0
best_k = 0
k_list = []
test_accuracy_list = []

for k in range(1, 382, 5):
    for j in range(10):
        # 划分数据集，training set：test set = 9:1
        testSet = np.vstack((dataSet[270 - 30 * j:300 - 30 * j], dataSet[480 - 20 * j:500 - 20 * j]))
        trainingSet = np.vstack((dataSet[0:270 - 30 * j], dataSet[300 - 30 * j:300], dataSet[300:480 - 20 * j], dataSet[500 - 20 * j:500]))
        testLabels = labels[270 - 30 * j:300 - 30 * j] + labels[480 - 20 * j:500 - 20 * j]
        trainingLabels = labels[0:270 - 30 * j] + labels[300 - 30 * j:300] + labels[300:480 - 20 * j] + labels[500 - 20 * j:500]
        for i in range(50):
            outputLabels.append(classify0(testSet[i],trainingSet,trainingLabels,k))
            # if classify0(validationSet[i],trainingSet,trainingLabels,k) == 'A':
            #   aLabelNumber += 1
            if outputLabels[i] == testLabels[i]:
                correctCount += 1

        correctCountSum += correctCount
        correctCount = 0
        outputLabels = []

    test_accuracy = correctCountSum/500
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_k = k

    k_list.append(k)
    test_accuracy_list.append(test_accuracy)
    print('k = ', k, ', test accuracy is :', test_accuracy)
    correctCountSum = 0

print('best k = ', best_k, ', best test accuracy is :',best_accuracy)

# 绘制曲线
plt.figure()
names = k_list
x = range(len(names))
y = test_accuracy_list
plt.plot(x, y, marker='o', mec='r', mfc='w')
plt.xticks(x, names, rotation=45)
plt.xlabel(u"k") #X轴标签
plt.ylabel("test accuracy") #Y轴标签
plt.title("不同k下的test accuracy 曲线图")


# 区域分割
plt.figure()
x_train = dataSet
y_train = labels
model = neighbors.KNeighborsClassifier(n_neighbors=best_k, weights='distance')# 使用之前获得的best k值
model.fit(x_train, y_train)

x_min, y_min = x_train.min(axis=0)
x_max, y_max = x_train.max(axis=0)
# np.linspace(x_min, x_max, 500)在[x_min,x_max]中产生500个均匀间隔的数字（包括尾部）
t1 = np.linspace(x_min, x_max, 500)
t2 = np.linspace(y_min, y_max, 500)
xx, yy = np.meshgrid(t1, t2)  # 生成网格采样点

grid_test=np.stack((xx.flat, yy.flat), axis=1) #测试点  （xx.flat降维）
y_predict = model.predict(grid_test)

mpl.rcParams['axes.unicode_minus'] = False #不然坐标下的编号会出现乱码
# 核心绘图函数
cm_bg = mpl.colors.ListedColormap(['r', 'y']) #背景颜色（样本分为2个类，所以为两个颜色）＃red代表A点区域，yellow代表B点区域
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max) #设置坐标范围

array = y_predict.reshape(xx.shape)
array = array.astype(float)
plt.pcolormesh(xx, yy, array, cmap=cm_bg) #绘制网格背景
#plt.pcolormesh()会根据y_predict的结果自动的在cmap中选择颜色
#plt.scatter(x_train[:,0],x_train[:,1],c='r',cmap=cm_pt,marker='o',edgecolors='k') #绘制样本点
plt.scatter(x1, y1, c = 'k', marker = '.')
plt.scatter(x2, y2, c = 'b', marker = 'x')
#plt.scatter()会根据y_train的结果自动的在cmap中选择颜色，c参数代表颜色
plt.title("决策分类图")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

