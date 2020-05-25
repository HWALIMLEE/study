#counter는 연속된 값을 defaultdict(int)와 유사한 객체로 변환해주며, 키와 값의 빈도를 연결시켜 준다.
from collections import Counter
c=Counter([0,1,2,0]) #{0:2,1:1,2:1} 0이 2개, 1이 1개, 2가 1개
print(c)

#특정 문서에서 단어의 개수를 셀 때도 유용
word_counts=Counter(document)

#가장 자주 나오는 단어 10개와 이 단어들의 빈도수 출력
for word, count in word_counts.most_common(10):
    print(word,count)
