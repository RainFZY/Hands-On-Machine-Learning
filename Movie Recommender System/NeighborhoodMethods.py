from math import sqrt
# Collaborative Filtering：不需要知道movie的genres，但需要事先知道被推荐玩家的打分
# Neighborhood Methods: 核心思想是找出和某用户喜好最接近的几位用户，根据他们的喜好来推荐给该玩家
# 一个涉及影评者及其几部影片评分情况的字典
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

# 新建一个相同的字典，用来后续将某个用户对某部电影的评分除以他的平均打分，对每个评分重新赋值
ave_critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snake on a Plane': 3.5,
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


# 计算用户之间的相近度，client1是被推荐的用户
def distance(client1,client2):
    sum = 0
    #计算所有距离的和（差值的平方和）
    #if they have no rating in common, return 0
    for item in critics[client1]:
        if item in critics[client2]:
            sum += pow(critics[client1][item] - critics[client2][item], 2)
    # sum越大，差异越大
    # 为了与用户相近度呈正相关，取欧氏距离的倒数，分母加一是防止分母为0的情况
    return 1/(1+sqrt(sum))

# 根据相似度排序用户，这个函数可以不用到
def rankDistance(client,n=6):
    rank_list = []
    for item in critics:
        if item != client:
            rank_list.append((distance(client,item),item))
    #排序
    rank_list.sort()
    rank_list.reverse()
    return rank_list[0:n]

# 计算综合评分、综合推荐指数
def getRecommendations(client):
    rec_dic = {} # 存放推荐的电影和推荐指数
    movieNum_dic = {} # 存放推荐的电影在其他用户中出现的总次数
    # 计算综合评分
    for other_client in ave_critics:
        movieNum = 0 # 统计某个用户评价过的电影总数
        scoreSum = 0 # 统计某个用户评价过的所有电影总分数
        if other_client != client:
            for movie in ave_critics[other_client]:
                movieNum += 1
                scoreSum += ave_critics[other_client][movie]

            averageScore = scoreSum/movieNum # 某个用户对电影的平均打分
            for movie in ave_critics[other_client]:
                # 将每个用户对某部电影的评分除以他的平均打分
                ave_critics[other_client][movie] = ave_critics[other_client][movie]/averageScore
                # 把符合被推荐资格的电影放入推荐字典中
                if movie not in ave_critics[client] and movie not in rec_dic:
                    rec_dic[movie] = 0
                    movieNum_dic[movie] = 0
    # print("计算平均后的用户-电影字典：",ave_critics)
    # 计算被推荐电影的综合推荐指数
    for other_client in ave_critics:
        if other_client != client:
            for movie in ave_critics[other_client]:
                if movie in rec_dic:
                    # 用户近似度 * 用户平均评分
                    rec_dic[movie] += distance(client,other_client) * ave_critics[other_client][movie]
                    movieNum_dic[movie] += 1
    for movie in rec_dic:
        rec_dic[movie] /= movieNum_dic[movie]
    print("推荐电影被其他用户评分的次数：",movieNum_dic)
    print("推荐电影综合推荐指数：",rec_dic)
    print("推荐电影顺序：",sorted(rec_dic.keys(), reverse=True))


print("用户相近度排序：",rankDistance('Toby',6))
getRecommendations('Toby')

