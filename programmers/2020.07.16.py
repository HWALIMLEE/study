# 행렬의 덧셈
def solution(arr1, arr2):
    answer = []
    for i, k in zip(arr1, arr2):
         answer.append([x+y for x,y in zip(i,k)])
    return answer

# 다른풀이
import numpy as np
def solution(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    answer = arr1 + arr2
    return answer.tolist()


# x만큼 간격이 있는 n개의 숫자
def solution(x, n):
    answer=[]
    if x != 0:
        return list(range(x,x*(n+1),x))
    else: # x가 0일때
        for i in range(0,n):
            answer.append(0)
        return answer

# 다른 풀이
def solution(x, n):
    answer = [i*x for i in range(1,n+1)]
    return answer