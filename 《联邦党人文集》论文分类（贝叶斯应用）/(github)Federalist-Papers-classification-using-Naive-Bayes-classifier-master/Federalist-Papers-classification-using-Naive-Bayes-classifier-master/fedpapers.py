from __future__ import division
import numpy as np
import json
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer



x = open('fedpapers_split.txt').read()
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


nH, nM, nD = len(papersH), len(papersM), len(papersD)
# 分别是51, 15, 11。还有3篇是Hamilton和Madison一起写的，还有5篇是Jay写的



# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results

# Stop words are words like “and”, “the”, “him”, which are presumed
# to be uninformative in representing the content of a text, and which may
# be removed to avoid them being construed as signal for prediction.
stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})
#stop_words = {'HAMILTON','MADISON'}

## Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words=stop_words,min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

#特征词列表，共1254个词
feature_names_list = vectorizer.get_feature_names()
print(feature_names_list)
print('total feature words number is :',len(feature_names_list))
# X：array, [n_samples, n_features]，每一行是每一篇文章，每一列是每一个特征词
# Document-term matrix.
# print(X)

# Uncomment this line to see the full list of words remaining after filtering out 
# stop words and words used less than min_df times
#vectorizer.vocabulary_

# Split word counts into separate matrices
# 分隔出每一类的array
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]
'''
print('---------------------')
print(XH)
print('---------------------')
print(XM)
print('---------------------')
print(XD)
'''

# Estimate probability of each word in vocabulary being used by Hamilton，存在列表fH中
fH = []
k = XH.sum(axis=0) #求每一列的和，即Hamilton所有文章中某个特征词出现的次数，一行1254列
print(k)
total_sum = sum(k)
# print(sum(k))
# 依个
for i in range(0,len(feature_names_list)):
    prob = ((k[i] + 1)/(float(total_sum + len(feature_names_list))))
    fH.append(prob)

print(fH)
# print(len(fH))

# Estimate probability of each word in vocabulary being used by Madison
fM = []
k = XM.sum(axis=0)
total_sum = sum(k)

for i in range(0,len(XM[1])):
    prob = ((k[i] + 1)/float(total_sum + len(XM[1])))
    fM.append(prob)
print(fM)

# Compute ratio of these probabilities
#fratio = fH/fM
fratio = [a/b for a,b in zip(fH,fM)]
print(fratio)

# Compute prior probabilities 先验概率

piH = len(XH)/float(len(X)) # 51/77
piM = len(XM)/float(len(X)) # 15/77


print('共',nD,'篇未知文章,文章编号及预测作者分别是：')
LR_List = []
for xd in range(nD): # Iterate over disputed documents, nD = 11
    # Compute likelihood ratio for Naive Bayes model
    # XD[xd]是一个由0,1组成的列表，1行1254列
    # np.power(a,b)：a的b次方，a是fratio（每个词的概率比），b是XD[xd]中的每个元素（0或1）
    # 所以该篇文章中没出现的词它的概率就记为1，不影响之后的乘积结果
    tmp = [np.power(a,b) for a,b in zip(fratio,XD[xd])]
    # print('------------------------------------------------------')
    # 每篇不确定文章中所有出现的特征词概率比（H/M）的乘积
    tmp = np.prod(np.array(tmp))
    # print(tmp)
    # LR：P(该篇文章是Hamilton写的)/P(该篇文章是Madison写的)的概率估计
    LR = tmp*(piH)/(piM)
    LR_List.append(LR)
    if LR>1:
        print('No',papersDNum[xd],': Hamilton')
    else:
        print('No',papersDNum[xd],': Madison')

print(LR_List[7])