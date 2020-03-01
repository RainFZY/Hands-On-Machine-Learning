from numpy import *
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# 从.txt文件中读取数据
def loadData():
    '''
    inFile = open(flieName, 'r')  # 以只读方式打开某fileName文件

    # 定义两个空list，用来存放文件中的数据
    x = []
    y = []

    for line in inFile:
        trainingSet = line.split(',')  # 对于每一行，按','把数据分开，这里是分成两部分
        x.append(trainingSet[0])  # 第一部分，即文件中的第一列数据逐一添加到list x 中
        y.append(trainingSet[1])  # 第二部分，即文件中的第二列数据逐一添加到list y 中

    '''
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
        labels.append('A')

    for i in range(200):
        labels.append('B')


    return np.round(data, 4),labels  # x,y组成一个元组，这样可以通过函数一次性返回


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

# 划分数据集，training:validation:test = 8:1:1，
dataSet,labels = loadData()

trainingSet = np.vstack((dataSet[0:240],dataSet[300:460]))
validationSet = np.vstack((dataSet[240:270], dataSet[460:480]))
testSet = np.vstack((dataSet[270:300], dataSet[480:500]))

trainingLabels = labels[0:240] + labels[300:460]
validationLabels = labels[240:270] + labels[460:480]
testLabels = labels[270:300] + labels[480:500]


x1, y1 = dataSet[0:240].T # 训练集中的A类点
x2, y2 = dataSet[300:460].T # 训练集中的B类点
plt.scatter(x1, y1, c = 'r', marker = 'o')
plt.scatter(x2, y2, c = 'b', marker = 'x')
plt.axis()
plt.title("trainingSet_ab_distribution")
plt.xlabel("x")
plt.ylabel("y")


plt.figure() # 为第二张图新建一个窗口
x1, y1 = dataSet[240:270].T # 校验集中的A类点
x2, y2 = dataSet[460:400].T # 校验集中的B类点
plt.scatter(x1, y1, c = 'r', marker = 'o')
plt.scatter(x2, y2, c = 'b', marker = 'x')
plt.axis()
plt.title("validationSet_ab_distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show() # plt.show()加在最后，使运行时能同时显示多张图片


outputLabels = []
correctCount = 0
aLabelNumber = 0
validation_accuracy = 0
best_accuracy = 0
best_k = 0
'''
for k in range(1, 52, 1):
    for i in range(500):
        outputLabels.append(classify0(validationSet[i],trainingSet,trainingLabels,k))
        # if classify0(validationSet[i],trainingSet,trainingLabels,k) == 'A':
        #   aLabelNumber += 1

        if outputLabels[i] == validationLabels[i]:
            correctCount += 1

    #print(aLabelNumber)
    #print(validationLabels)
    #print(outputLabels)
    validation_Accuracy = correctCount/500
    #print('k = ', k, ', validation accuracy is :',validation_Accuracy)
    outputLabels = []
    correctCount = 0
    aLabelNumber = 0
'''

for k in range(1, 322, 5):
    for i in range(50):
        outputLabels.append(classify0(validationSet[i],trainingSet,trainingLabels,k))
        # if classify0(validationSet[i],trainingSet,trainingLabels,k) == 'A':
        #   aLabelNumber += 1

        if outputLabels[i] == validationLabels[i]:
            correctCount += 1

    #print(aLabelNumber)
    #print(validationLabels)
    #print(outputLabels)
    validation_accuracy = correctCount/50
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_k = k

    print('k = ', k, ', validation accuracy is :',validation_accuracy)
    outputLabels = []
    correctCount = 0
    aLabelNumber = 0

print('best k = ', best_k, ', validation accuracy is :',best_accuracy)

# ------------------------------------测试---------------------------------------
testCorrectNum = 0
for i in range(50):
    outputLabels.append(classify0(testSet[i],trainingSet,trainingLabels,best_k))
    # if classify0(validationSet[i],trainingSet,trainingLabels,k) == 'A':
    #   aLabelNumber += 1

    if outputLabels[i] == testLabels[i]:
        testCorrectNum += 1


test_Accuracy = testCorrectNum/50
print('k = ', best_k, ', test accuracy is :',test_Accuracy)



'''
    testX = array([0, 0])
    testY = array([1,2])
    k = 3
    outputLabelX = classify0(testX,dataSet,labels,k)
    outputLabelY = classify0(testY,dataSet,labels,k)

print('input is :',testX,'output class is :',outputLabelX)
print('input is :',testY,'output class is :',outputLabelY)
'''