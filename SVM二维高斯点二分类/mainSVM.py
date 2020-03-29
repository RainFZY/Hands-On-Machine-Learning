from pylab import *
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import GridSearchCV

#消除警告
warnings.filterwarnings("ignore")

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
        labels.append(0) # 1代表A点
    for i in range(200):
        labels.append(1) #　２代表Ｂ点

    # 保留四位小数
    return np.round(data, 4),labels


# 划分数据集
dataSet,labels = createData()

trainingSet = np.vstack((dataSet[0:240],dataSet[300:460]))
testSet = np.vstack((dataSet[240:300], dataSet[460:500]))
trainingLabels = labels[0:240] + labels[300:460]
testLabels = labels[240:300] + labels[460:500]

x1, y1 = dataSet[0:300].T # 所有A类点
x2, y2 = dataSet[300:500].T # 所有B类点

# 绘制生成的ab点分布图
def plotPoints(dataset):
    plt.scatter(x1, y1, c = 'r', marker = '.')
    plt.scatter(x2, y2, c = 'b', marker = 'x')
    plt.axis()
    plt.title("A&B points distribution")
    plt.xlabel("x")
    plt.ylabel("y")

# 线性核
# C：错误项的惩罚系数，C越大泛化能力越弱，越容易过拟合，C跟松弛向量有关
parameters = {'C': [0.5, 1, 3, 5, 7, 9]} #
clf1 = GridSearchCV(SVC(kernel='linear'), parameters, scoring='f1') # 选择最佳参数
clf1.fit(trainingSet, trainingLabels)  # 训练
print('best parameters of linear kernel:',clf1.best_params_)
clf1 = SVC(kernel='linear', C=clf1.best_params_['C'])
clf1.fit(trainingSet, trainingLabels)

# 高斯核
# gamma：核函数系数
parameters = {'C': [0.5, 1, 3, 5, 7, 9], 'gamma': [0.001, 0.125, 0.25, 0.5, 1, 2, 4]}
clf2 = GridSearchCV(SVC(kernel='rbf'), parameters, scoring='f1') # 选择最佳参数
clf2.fit(trainingSet, trainingLabels)
print('best parameters of gaussian kernel:',clf2.best_params_)
clf2 = SVC(kernel='rbf',C=clf2.best_params_['C'],gamma=clf2.best_params_['gamma'])
clf2.fit(trainingSet, trainingLabels)

# 计算预测准确率
p1 = 0  # 正确分类的个数
p2 = 0
for j in range(len(testSet)):  # 循环检测测试数据分类成功的个数
    if clf1.predict(np.array([testSet[j]])) == testLabels[j]:
        p1 += 1
    if clf2.predict(np.array([testSet[j]])) == testLabels[j]:
        p2 += 1

print("accuracy of linear kernel is:",p1,"%") # 输出测试集准确率
print('accuracy of rbf kernel is: ',p2,"%")

# 绘制用linear核的SVM得到的超平面图
def plot_linear_hyperplane(clf, title='hyperplane'):
    w = clf.coef_[0] # 取的w的值
    a = -w[0]/w[1] # 点斜式的斜率
    xx = np.linspace(-3, 4) # 从-3到4产生连续的值
    yy = a*xx - (clf.intercept_[0])/w[1]#clf.intercept_[0]相当于是b或w3
    # 写成y=a*x+b形式: w_0*x + w_1*y +w_3=0能被改写成 y = -(w_0/w_1) x + (w_3/w_1)

    # 绘制过支持向量的两条虚线
    b = clf.support_vectors_[0]
    yy_down = a*xx + (b[1] - a*b[0]) # b[0]和b[1]分别是支持向量的横纵坐标
    b = clf.support_vectors_[-1] # -1指的是最后一个值
    yy_up = a*xx + (b[1] - a*b[0])

    # 绘图
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf1.support_vectors_[:, 0], clf1.support_vectors_[:, 1],s=80, facecolors='y')
    plt.scatter(x1, y1, c = 'r', marker = '.')
    plt.scatter(x2, y2, c = 'b', marker = 'x')
    plt.axis('tight')


# 绘制用高斯核的SVM得到的超平面图
def plot_gaussian_hyperplane(clf, X,title='hyperplane'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)#

    plt.scatter(clf1.support_vectors_[:, 0], clf1.support_vectors_[:, 1], s=80, facecolors='y')
    plt.scatter(x1, y1, c='r', marker='.')
    plt.scatter(x2, y2, c='b', marker='x')
    plt.axis('tight')


# 调用函数绘图
plt.figure()
plotPoints(dataSet)
plt.figure()
plot_linear_hyperplane(clf1, title='linear kernel hyperplane')
plt.figure()
plot_gaussian_hyperplane(clf2, dataSet, title='gaussian kernel hyperplane')
plt.show()







