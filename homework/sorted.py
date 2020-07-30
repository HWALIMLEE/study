nest_list=[
    [0,9],
    [1,8],
    [2,7],
    [3,6],
    [4,5],
]

print(sorted(nest_list,key=lambda x:x[1]))

#문제
#"시"가 오름차순이 되도록 time_data를 정렬해서 출력하시오
time_data=[
    [2006,11,26,2,40],
    [2009,1,16,23,35],
    [2014,5,4,14,26],
    [2017,8,9,7,5],
    [2020,1,5,22,15]
]

time_data=sorted(time_data,key=lambda x:x[3])
print(time_data)

