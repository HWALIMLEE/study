#sort메서드-기존의 정렬을 새롭게 바꿈
#sorted메서드-기존의 정렬 유지하고 새롭게 정렬된 것 반환

#오름차순이 default, 내림차순으로 변경하고 싶다면 reverse=True써줄 것!
x=sorted([-4,1,-2,3],key=abs,reverse=True) #절댓값 기준으로 내림차순!
print(x) #[-4,3,-2,1]

#리스트이 각 항목끼리 서로 비교하는 대신 key를 사용하면 지정한 함수의 결과값을 기준으로 리스트 정렬!
# wc=sorted(word_counts.items(),key=lambda word_and_count:word_and_count[1],reverse=True)

#좀 더 쉬운 예제
x={1:'a',3:'d',4:'c',2:'b',0:'e'}
sorted_list=sorted(x)
for y in sorted(x):
    print(y,x[y])


print("================")
#key lambda
for y, v in sorted(x.items(),key=lambda x:x[1]):
    print(y,v)

# key=lambda x:x[0]--->key값으로 정렬
# key=lambda x:x[1]--->value값으로 정렬

