from __future__ import division
import numpy as np
import json
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer


x = open('split_papers.txt').read()
papers = json.loads(x) # papers是一个列表
# print(papers[0][0][105])
# papers[0][0]就是Hamilton的第一篇文章
# papers[0][0][105]就是Hamilton的第一篇文章的第106个字符
papersH = papers[0] # papers by Hamilton 
papersM = papers[1] # papers by Madison
papersD = papers[2] # disputed papers

# 把未知文章的篇数号加到一个列表中
papersDNum = []
for i in range(11):
    papersDNum.append(papersD[i][0]+papersD[i][1])


numH, numM, numD = len(papersH), len(papersM), len(papersD)
# 分别是51, 15, 11。还有3篇是Hamilton和Madison一起写的，还有5篇是Jay写的

# 摘自官方文档：
# Stop words are words like “and”, “the”, “him”, which are presumed
# to be uninformative in representing the content of a text, and which may
# be removed to avoid them being construed as signal for prediction.
stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})


# 官方文档：https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
# min_df表示最少出现次数，用来筛选特征词，可改
vectorizer = text.CountVectorizer(stop_words = stop_words, min_df = 10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

#特征词列表，共1254个词
feature_names_list = vectorizer.get_feature_names()
print('Feature words are:',feature_names_list)
print('Total feature words number is :',len(feature_names_list))
# X：array, [n_samples, n_features]，每一行是每一篇文章，每一列是每一个特征词
# Document-term matrix.
# print(X)

# Split word counts into separate matrices
# 分隔出每一类的array
XH, XM, XD = X[:numH,:], X[numH:numH+numM,:], X[numH+numM:,:]
'''
print('---------------------')
print(XH)
print('---------------------')
print(XM)
print('---------------------')
print(XD)
'''

# 估计Hamilton所有文章中各个特征词出现的概率，存在列表H_fw_prob中
H_fw_prob = []
k = XH.sum(axis=0) #求每一列的和，即Hamilton所有文章中某个特征词出现的次数，一行1254列
print('Hamilton所有文章中各特征词出现的总次数：',k)
total_sum = sum(k)
print('Hamilton所有文章所有特征词出现的总次数：',sum(k))
for i in range(0,len(feature_names_list)):
    # 为了防止k[i]=0，导致概率为0，分子加一，分母加上len(XM[1])(1254)
    prob = ((k[i] + 1)/float(total_sum + len(XH[1])))
    H_fw_prob.append(prob)
print('Hamilton所有文章中各特征词出现的频率：',H_fw_prob)


# 估计Madison所有文章中各个特征词出现的概率，存在列表M_fw_prob中
M_fw_prob = []
k = XM.sum(axis=0)
print('Madison所有文章中各特征词出现的总次数：',k)
total_sum = sum(k)
print('Madison所有文章所有特征词出现的总次数：',sum(k))
for i in range(0,len(XM[1])):
    # 为了防止k[i]=0，导致概率为0，分子加一，分母加上len(XM[1])(1254)
    prob = ((k[i] + 1)/float(total_sum + len(XM[1])))
    M_fw_prob.append(prob)
print('Madison所有文章中各特征词出现的频率：',M_fw_prob)



# Compute ratio of these probabilities
prob_ratio = [a/b for a,b in zip(H_fw_prob,M_fw_prob)]
print('各特征词在两个作者文章中出现的频率比：',prob_ratio)

# Compute prior probabilities 先验概率
probH = len(XH)/float(len(X)) # 51/77
probM = len(XM)/float(len(X)) # 15/77


print('共',numD,'篇争议文章,文章编号及预测作者分别是：')
prob_HM_List = []
for xd in range(numD): # numD = 11
    # Compute likelihood ratio for Naive Bayes model
    # XD[xd]是一个由0,1组成的列表，1行1254列
    # np.power(a,b)：a的b次方，a是prob_ratio（每个词的概率比），b是XD[xd]中的每个元素（0或1）
    # 所以该篇文章中没出现的词它的概率就记为1（0次方），不影响之后的乘积结果
    power = [np.power(a,b) for a,b in zip(prob_ratio,XD[xd])]
    # print('------------------------------------------------------')
    # 每篇不确定文章中所有出现的特征词概率比（H/M）的乘积
    mul = np.prod(np.array(power))
    # print(tmp)
    # prob_HM：P(该篇文章是Hamilton写的)/P(该篇文章是Madison写的)的概率估计
    prob_HM = mul * (probH)/(probM)
    prob_HM_List.append(prob_HM)
    if prob_HM>1:
        print('No',papersDNum[xd],': Hamilton')
    else:
        print('No',papersDNum[xd],': Madison')

# print(prob_HM_List[7])