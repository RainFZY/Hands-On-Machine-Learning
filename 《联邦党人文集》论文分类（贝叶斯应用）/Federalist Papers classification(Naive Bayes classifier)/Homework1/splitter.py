import io, json

# The Federalist Papers.txt是原始文集文件
text = open('The Federalist Papers.txt').readlines()
text = ''.join(text)

# 把原始文集文件按指定内容分隔开
split_text = text.split('FEDERALIST No. ')[1:]
split_text[84] = split_text[84].split("End of the Project Gutenberg EBook of The Federalist Papers, by")[0]

# 把85篇文章分为5类，对应不同的五种作者类型
Hamilton = []
Jay = []
Madison = []
Disputed = []
Hamilton_and_Madison = []


i = 0
while i < 85:
    tex = split_text[i]
    # 两位数时有两个字符位，所以要加上后半部分，不然10会识别为1
    number = int(tex.split('\r\n\r\n')[0][0]+tex.split('\r\n\r\n')[0][1])
    #print(number)
  
    if number >=62 and number <= 63:
        Disputed.append(tex)
    elif number >= 49 and number <=57:
        Disputed.append(tex)
    elif number >=18 and number <=20:
        Hamilton_and_Madison.append(tex)
    else:
        if 'HAMILTON' in tex:
            Hamilton.append(tex)
        elif 'JAY' in tex:
            Jay.append(tex)
        elif 'MADISON' in tex:
            Madison.append(tex)
    i+=1
print('作者是Hamilton的文章数量：',len(Hamilton))
print('作者是Madison的文章数量：',len(Madison))
print('作者是Hamilton和Madison的文章数量：',len(Hamilton_and_Madison))
print('作者是Jay的文章数量：',len(Jay))
print('作者有争议的文章数量：',len(Disputed))
# 新建一个列表，来存放我我们感兴趣的的三类作者文章
papers = []
papers.append(Hamilton)
papers.append(Madison)
papers.append(Disputed)
# print(papers)

# 写入文件保存
with open('split_papers.txt', 'w') as outfile:
  json.dump(papers, outfile) # json.dumps将一个Python数据结构转换为JSON


