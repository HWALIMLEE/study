#고차함수
#리스트이 각 요소에 함수를 적용하려면 map()함수 이용

# for루프
a=[1,-2,3,-4,5]
new=[]
for x in a:
    new.append(abs(x))
print(new)

# map루프
a=[1,-2,3,-4,5]
a_answer=list(map(abs,a))
print("a_answer:",a_answer)

import re
time_list=[
    "2006/11/26_2:40",
    "2009/1/16_23:35",
    "2014/5/4_14:26",
    "2017/8/9_7:5",
    "2020/1/8_22:15"
]

# 문자열에서 "시"를 추출
get_hour=lambda x: int(re.split("[/_:]",x)[3])
hour_list=list(map(get_hour,time_list))
print("get_answer:",hour_list)

