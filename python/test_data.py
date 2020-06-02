#날짜 거꾸로, 앙상블 모형, 소스,npy,h5제출
import pandas as pd
import numpy as np
from keras.models import Input,Model
from keras.layers import Dense, LSTM
import random

samsung=pd.read_csv("./data/csv/samsung.csv",index_col=0,header=0,sep=',',encoding='CP949') 
hite=pd.read_csv("./data/csv/hite.csv",index_col=0,header=0,sep=',',encoding='CP949')
print(samsung.shape) # (509,1)
print(hite.shape) # (509,5) 

for i in range(len(samsung.index)):
    samsung.iloc[i,0]=int(samsung.iloc[i,0].replace(',',''))

#결측치 제거
hite_drop=hite.dropna(axis=0)


for i in range(len(hite_drop.index)):
    for j in range(len(hite_drop.iloc[i])):
            if type(hite_drop.iloc[i,j])==str:
                hite_drop.iloc[i,j]=int(hite_drop.iloc[i,j].replace(',',''))

print(hite_drop.head())

hite=hite_drop
samsung=samsung.sort_values(['일자'], ascending=True)
hite=hite.sort_values(['일자'],ascending=True)

print(samsung.head())
print(hite.head())

samsung=samsung.values
hite=hite.values

np.save('./data/samsung.npy',arr=samsung)
np.save('./data/hite.npy',arr=hite_drop)
