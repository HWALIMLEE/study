#1
word_counts={}
for word in document:
    if word in word_counts:
        word_counts[word]+=1
    else:
        word_counts[word]=1
#2.
word_counts={}
for word in document:
    try:
        word_counts[word]+=1
    except KeyError:
        word_counts[word]=1

#3. 존재하지 않는 키 적절하게 처리해주는 get사용
word_counts={}
for word in document:
    previous_count=word_counts.get(word,0)
    word_counts[word]=previous_count+1

#세가지 방법 모두 복잡
#defaultdict 사용하면 편리해진다.
from collections import defaultdict

word_counts=defaultdict(int)
for word in document:
    word_counts[word]+=1


dd_pair=defaultdict(lambda:[0,0])
dd_pair[2][1]=1
print(dd_pair)