import io, json

text = open('The Federalist Papers.txt').readlines()
text = ''.join(text)

split_text = text.split('FEDERALIST No. ')[1:]

split_text[84] = split_text[84].split("End of the Project Gutenberg EBook of The Federalist Papers, by")[0]

# Splits the essays into Disputed, Madison, Hamilton, and Jay categories.
Hamilton = []
Jay = []
Madison = []
Disputed = []
Hamilton_and_Madison = []



i = 0
while i < 85:
  tex = split_text[i]

  number = int(tex.split('\r\n\r\n')[0][0]+tex.split('\r\n\r\n')[0][1]) # 两位数时有两个字符位，所以要加上后半部分，不然10会识别为1
  #print(number)
  
  if number >=62 and number <= 63:
    Disputed.append(tex)
  elif number >= 49 and number <=57:
    Disputed.append(tex)
  elif number >=18 and number <=20: # HAMILTON AND MADISON
    Hamilton_and_Madison.append(tex)
  else:
    if 'HAMILTON' in tex:
      Hamilton.append(tex)
    elif 'JAY' in tex:
      Jay.append(tex)
    elif 'MADISON' in tex:
      Madison.append(tex)
  i+=1

papers = {'Hamilton':Hamilton, 'Jay':Jay, 'Madison':Madison, 'Disputed':Disputed, 'Hamilton_and_Madison':Hamilton_and_Madison}

# 比如，papers['Hamilton'][0]就是Hamilton的第一篇文章，papers['Hamilton'][50]就是Hamilton的第一篇文章，Hamilton共51篇文章
for k in papers.keys():
  print(k)
  print(len(papers[k]))


with open('papers.txt', 'w') as outfile:
  json.dump(papers, outfile) # json.dumps将一个Python数据结构转换为JSON


