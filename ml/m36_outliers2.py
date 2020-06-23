import numpy as np

# 컬럼별로 분리
def outliers(data_out):
    out = []
    for col in range(data_out.shape[1]):
        columns = data_out[:,col]
        quartile_1, quartile_3 = np.percentile(columns,[25,75])
        print("제 1사분위 수:",quartile_1)
        print("제 3사분위 수:",quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        answer = np.where((columns > upper_bound)| (columns < lower_bound))
        out.append(answer)
    return out
        
        # "|" = or
    
    # np.where 조건에 맞는 색인값

# a = np.array([[1,2,3,4,10000,6,7,5000,90,100],[1,5,6,100,1000,500,8,9,10,15]])

a = np.array([[1,2],[3,4],[10000,6],[7,5000],[90,100],[1,2],[3,4],[10000,6],[7,5000],[90,100]])

b = outliers(a)

print("이상치의 위치:",b)
