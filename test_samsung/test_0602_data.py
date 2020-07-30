import numpy as np
import pandas as pd

samsung=pd.read_csv('./data/csv/samsung.csv',index_col=0,header=0,sep=',',encoding='cp949')  #날짜는 그냥 인덱스

hite=pd.read_csv('./data/csv/hite.csv',index_col=0,header=0,sep=',',encoding='cp949')  #header와 index빼주기
#index_col=None을 해버리면 일자가 데이터로 잡힘, header=None으로 하면 header가 일자, 시간 써져 있는 부분이 데이터로 잡힘(index와 header가 없다고 인식)

print("samsung.shape:",samsung.shape)

# None제거 1
samsung=samsung.dropna(axis=0) #axis=0은 행, axis=1은 열 /default은 행제거
hite=hite.fillna(method='bfill') #전날값으로 채운다/ 위의 행으로 채움
# # hite=hite.dropna(axis=0)
# print(hite.head())

# None제거 2
hite=hite[0:509]
#1.hite.iloc[0,1:5]=[10,20,30,40] #iloc의 i는 index
# hite.loc["2020-06-02","고가":"거래량"]=['30','40','50','60'] #콤마 꼭 넣어주어야 함


# None을 predict값으로 대체해보기(제일 좋은 방법)
# 연속값으로 생기게 됨
# 간단한 머신러닝으로 예측가능(xgboost, randomforest)

#오름차순 정렬
samsung=samsung.sort_values(['일자'],ascending=True)
hite=hite.sort_values(['일자'],ascending=True)

#콤마제거, 문자를 정수로 형변환
for i in range(len(samsung.index)):
    samsung.iloc[i,0]=int(samsung.iloc[i,0].replace(',',''))
print(samsung)
print(type(samsung.iloc[0,0])) #<class:'int'>

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])): #각 행마다 열의 개수만큼
        hite.iloc[i,j]=int(hite.iloc[i,j].replace(',',''))

print(hite)
print(type(hite.iloc[1,1]))

np.save('./data/samsung.npy',arr=samsung)
np.save('./data/hite.npy',arr=hite)