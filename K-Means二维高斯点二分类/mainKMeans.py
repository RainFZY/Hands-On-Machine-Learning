from pylab import *
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import GridSearchCV

#消除警告
warnings.filterwarnings("ignore")
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False #不然坐标下的编号会出现乱码

def createData():
    # A点：(0,0)为中心、(1,0；0,1）为协方差矩阵的二维高斯分布
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean1, cov1, 300)
    # A点：(1,2)为中心、(1,0；0,2）为协方差矩阵的二维高斯分布
    mean2 = [1, 2]
    cov2 = [[1, 0], [0, 2]]
    data = np.append(data,np.random.multivariate_normal(mean2, cov2, 200),0)
    # print(np.round(data, 4))

    labels = []
    for i in range(300):
        labels.append(0) # 0代表A点
    for i in range(200):
        labels.append(1) # 1代表Ｂ点

    # 保留四位小数
    return np.round(data, 4),labels




# 绘制生成的原始ab点分布图
def plotPoints(dataSet):
    plt.scatter(dataSet[0:300].T[0],dataSet[0:300].T[1], c = 'r', marker = '.')
    plt.scatter(dataSet[300:500].T[0],dataSet[300:500].T[1], c = 'b', marker = 'x')
    plt.axis()
    plt.title("A&B points distribution")
    plt.xlabel("x")
    plt.ylabel("y")


# 欧氏距离计算
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

# 从最小最大范围内选初始点，非从数据集中选，未用到
def random_center(data, k):
    n = data.shape[1]
    centroids = np.mat(np.zeros((k, n)))

    for i in range(n):
        data_min = np.min(data[:, i])
        data_range = np.float(np.max(data[:, i]) - data_min)
        centroids[:, i] = np.mat(data_min + data_range + np.random.rand(k, 1))

    return centroids


# 生成k个随机类心
def randomCenters(dataSet, k):
    m, n = dataSet.shape # m = 500, n = 2
    clusterCenters = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  # 在0-500之中随机采样
        clusterCenters[i, :] = dataSet[index, :] # 数据集中的第index个点加入初始类心集合，共加入k个

    # clusterCenters是k*n矩阵，每一行是一个类心的点坐标
    return clusterCenters

# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0] # 行的数目也就是样本点的个数
    # pointsMat第一列存样本属于哪一簇,第二列存样本的到簇的中心点的误差，用距离的平方表示
    pointsMat = np.mat(np.ones((m, 2))*2) # 初始化一个全是2的矩阵，避免跟0、1混淆
    clusterChange = True
    # 第一步：随机方法初始化类心
    clusterCenters = randomCenters(dataSet, k)
    # clusterCenters = random_center(dataSet, k)
    # print(clusterCenters)
    aCost = []
    bCost = []
    count = 0
    # 对所有点遍历过程中，有任何一个点改变了聚类，就设置成True，将开启下一轮遍历
    while clusterChange:
        count += 1 # 迭代轮数
        clusterChange = False
        # 遍历所有的样本（行数），迭代
        for i in range(m):
            clusterDist = 1000
            clusterIndex = -1
            # 第二步：遍历所有的类心,找出最近的类心，归类
            for j in range(k):
                # 计算该样本点到类心的欧式距离
                distance = distEuclid(clusterCenters[j, :], dataSet[i, :])
                if distance < clusterDist:
                    clusterDist = distance
                    clusterIndex = j
            # 第三步：更新每一行样本所属的类
            if pointsMat[i, 0] != clusterIndex:
                clusterChange = True # 对所有点遍历过程中，有任何一个点改变了聚类，就设置成True，将开启下一轮遍历
                pointsMat[i, :] = clusterIndex, clusterDist ** 2
            # 第四步：更新类心
            cx = clusterCenters[clusterIndex][0] # 原类心横坐标
            cy = clusterCenters[clusterIndex][1]
            clusterPoints = dataSet[np.nonzero(pointsMat[:, 0].A == clusterIndex)[0]]  # 获取一个cluster中所有的点
            # np.nonzero返回括号中数组中非零元素的索引值数组
            clusterCenters[clusterIndex, :] = np.mean(clusterPoints, axis=0) # 对矩阵的行求均值
            # 把每次迭代类心点的偏移距离作为cost
            cost = sqrt((clusterCenters[clusterIndex][0] - cx) ** 2 + (clusterCenters[clusterIndex][1] - cy) ** 2)
            if (clusterIndex == 0):
                aCost.append(cost)
            else:
                bCost.append(cost)
        # 得到迭代完毕后最终的类心
        for j in range(k):
            clusterPoints = dataSet[np.nonzero(pointsMat[:, 0].A == j)[0]]  # 获取一个cluster中所有的点
            # np.nonzero返回括号中数组中非零元素的索引值数组
            clusterCenters[j, :] = np.mean(clusterPoints, axis=0)  # 对矩阵的行求均值

    print("类心坐标：",clusterCenters[0],clusterCenters[1])
    # print(aCost)
    # print(bCost)
    # print(len(aCost))
    # print(len(bCost))
    print("迭代轮数：%d轮，共%d次"%(count,count*500))
    return clusterCenters, pointsMat, aCost, bCost


# 类心偏移距离随迭代次数变化曲线图
def costCurve(costList):
    plt.figure()
    x = range(100)
    y = costList[0:100]
    plt.plot(x, y, marker='o', mec='r', mfc='w')
    plt.xlabel(u"迭代次数")  # X轴标签
    plt.ylabel("类心偏移距离")  # Y轴标签


# 绘制聚类图以及类心点
def showCluster(dataSet, labelSet, k, clusterCenters, pointsMat):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1
    # 调整标记点顺序，保证聚类点颜色、标志与原图一致
    if(clusterCenters[0][0]>clusterCenters[1][0]):
        mark = ['xb', '.r', '^g', '.k', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    else:
        mark = ['.r', 'xb', '^g', '.k', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    if k > len(mark):
        print("k值太大了")
        return 1

    plt.figure()
    correctNum = 0 # 分类正确的样本点个数
    # 绘制所有的样本
    for i in range(m):
        markIndex = int(pointsMat[i, 0]) # 第i个点所属的类别序号
        if((clusterCenters[0][0]<clusterCenters[1][0] and markIndex == labelSet[i])
                or(clusterCenters[0][0]>clusterCenters[1][0] and 1-markIndex == labelSet[i])):
            correctNum += 1
        plt.title('clustering result')
        plt.plot(dataSet[i, 0],dataSet[i, 1], mark[markIndex])

    accuracy = correctNum/500
    print('The accuracy is: ',accuracy*100,'%')
    mark = ['Dk', 'Dg', 'Dr', 'Db', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制类心点
    for i in range(k):
        plt.plot(clusterCenters[i, 0], clusterCenters[i, 1], mark[i])


# 生成数据集，标签集
dataSet,labels = createData()
plt.figure()
plotPoints(dataSet)

k = 2
clusterCenters, pointsMat, aCost, bCost = KMeans(dataSet, k)
showCluster(dataSet, labels, k, clusterCenters, pointsMat)
costCurve(aCost)
plt.title("A类点类心偏移距离随迭代次数变化曲线图")
costCurve(bCost)
plt.title("B类点类心偏移距离随迭代次数变化曲线图")
plt.show()