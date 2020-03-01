## **Logistic Regression Classification on Gaussian Distribution 2D Points**

Python3 implementation of logistic regression classification algorithm using gradient descent

**要求：**

有2类二维空间点，A类和B类。

A类点以（0，0）为中心、（1，0；0，1）为协方差矩阵的二维高斯分布；

B类点以（1，2）为中心、（1，0；0，2）为协方差矩阵的二维高斯分布；

随机生成300个A类点，200个B类点，并用Logistic Regression的方法进行分类（GD）。画出不同epoch等参数下的结果。

 

**思路：**

1. 生成数据集：首先生成这500个点，以列表的形式表示位置坐标，再把所有的坐标放入一个大的列表中。再生成一个label列表，包含300个’A’和200个’B’。

2. 为了更加清晰直观，把生成的300个A类点和200个B类点分别用两种颜色的符号绘制在一个二维坐标系内。

3. 引入逻辑回归算法，包括定义sigmoid函数、定义梯度上升算法函数来求出参数θ（由于数据样本较小，为保证更高的准确性，采用GD而不是SGD）。

4. 增加附加功能：根据画出来的决策边界计算准确率，绘制epoch-accuracy曲线，求得最高正确率以及最高正确率下的最小的迭代次数。

5. 利用求出来的参数θ以及最佳且最小迭代次数画出决策边界。



**结果：**

AB点分布图（是红色点、是蓝色叉）：

![ab_distribution](https://github.com/RainFZY/Hands-On-Machine-Learning/tree/master/逻辑回归二维高斯点二分类/ab_distribution.jpg)

不同迭代次数epoch下的准确率：

![results](https://github.com/RainFZY/Hands-On-Machine-Learning/tree/master/逻辑回归二维高斯点二分类/results.png)

迭代次数epoch与准确率曲线：

![epoch-accuracy-curve](https://github.com/RainFZY/Hands-On-Machine-Learning/tree/master/逻辑回归二维高斯点二分类/epoch-accuracy-curve.jpg)

可以看到，在迭代次数很小时，准确率随着迭代次数的增加有明显的提升。而当迭代次数达到30次左右时，准确率达到最高值并基本维持不变。最高准确率是88.2%，达到最高准确率的最小迭代次数是106次。

用最佳且最小迭代次数绘制的最佳拟合决策分界线：

![best-fit-line](https://github.com/RainFZY/Hands-On-Machine-Learning/tree/master/逻辑回归二维高斯点二分类/best-fit-line.jpg)