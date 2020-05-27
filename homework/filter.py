#리스트의 각 요소에서 조건에 맞는 요소만 꺼내고 싶을 땐 filter()함수 사용

#for 루프
a=[1,-2,3,-4,5]
new=[]
for x in a:
    if x>0:
        new.append(x)

#filte()함수의 예
a=[1,-2,3,-4,5]
print(list(filter(lambda x:x>0,a)))

#filter(조건이_되는_함수,배열)

#time_list에서 "월"이 1이상 6이하인 요소를 추출하여 배열 출력

import re
time_list=[
    "2006/11/26_2:40",
    "2009/1/16_23:35",
    "2014/5/4_14:26",
    "2017/8/9_7:5",
    "2020/1/8_22:15"
]

time_month=lambda x: int(re.split("[/_:]",x)[1])-7<0

print(list(filter(time_month,time_list)))
