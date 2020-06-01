import numpy as np
import pandas as pd
#numpy로 가져올 때 header 포함되어 있으면 오류가 난다(numpy는 한가지 자료형만 가능하기 때문

datasets=pd.read_csv("./data/csv/iris.csv",index_col=None,header=0,sep=',') #자동 인덱스 생성, header는 데이터 아님
print(datasets)
print(datasets.head()) #위에서부터 5개
print(datasets.tail()) #아래에서부터 5개

print(datasets.values) #데이터 값만 출력 
# .values는 pandas를 numpy로 바꾸는 것(한 가지 자료형으로)
print(type(datasets.values)) 
#numpy.ndarray
#header와 index를 제거하면 numpy로 쓸 수 있다

#넘파이로 저장
np.save('./data/iris.npy',arr=datasets)