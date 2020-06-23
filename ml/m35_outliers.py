import numpy as np

# 컬럼별로 분리할 수 있게 만들어야 함(지금은 1개짜리)---for문 들어가야함
def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out,[25,75])
    print("제 1사분위 수:",quartile_1)
    print("제 3사분위 수:",quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound)| (data_out < lower_bound))
    
    # np.where 조건에 맞는 색인값

a = np.array([1,2,3,4,10000,6,7,5000,90,100])

b = outliers(a)

print("이상치의 위치:",b)

# 실습: 행렬을 입력해서 행렬별로 이상치 발견하는 함수를 구현하시오.
# 파일명: m36_outliers2.py