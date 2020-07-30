# 정수 제곱근 판별하기
def solution(n):
    a = pow(n, 0.5)
    if a > int(a):
        return -1
    else:
        return pow((int(a)+1),2)

# 제일 작은 수 제거하기
def solution(arr):
    c = min(arr)
    import numpy as np
    if len(np.unique(arr)) > 1:
        arr.remove(c)
        return arr
    else:
        return [-1]
    