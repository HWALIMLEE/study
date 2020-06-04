#!/usr/bin/env python3
#coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from pandas import DataFrame
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

train=pd.read_csv("C:\\Users\\hwalim\\kaggle\\train.csv")
test=pd.read_csv("C:\\Users\\hwalim\\kaggle\\test.csv")

print("train.shape:",train.shape)
print("test.shape:",test.shape)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#age에 nan값이 존재
age_nan_rows=train[train['Age'].isnull()]

print(age_nan_rows.head())

##먼저 가장 간단한 성별을 0,1로 표시

from sklearn.preprocessing import LabelEncoder
train['Sex']=LabelEncoder().fit_transform(train['Sex'])
test['Sex']=LabelEncoder().fit_transform(test['Sex'])

print(train.head(10))

### 이름의 뒷부분을 고려하기엔 케이스가 너무 많아진다. 이름에서 앞의 성만 따서 생각
train['Name']=train['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
titles=train['Name'].unique()
titles
test['Name']=test['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip()) #(',')로 쪼갠 후 뒤에꺼, ('.')로 쪼갠 후 앞에꺼, 공백, 문자열 제거
test_titles=test['Name'].unique()
print("test_titles:",test_titles)

#성별로 나누는 것도 정확한 기준 부족, 해당 부분에 대해서 좀 더 생각해볼 필요

#fillna함수-NaN을 특정 값으로 대체하는 기능을 한다datetime A combination of a date and a time. Attributes: ()
#특정 텍스트, 평균값...
#inplace옵션에 true를 주면 또 다른 객체를 반환하지 않고, 기존 객체를 수정
train['Age'].fillna(-1,inplace=True) #-1값으로 대체
test['Age'].fillna(-1,inplace=True)

medians=dict() #dict()생성자는 key-value쌍을 갖는 tuple리스트를 받아들이거나
for title in titles:
    median=train.Age[(train["Age"]!=-1)]
    medians[title]=median

for index,row in train.iterrows(): #반복 처리(A-->B로 바꾸기/ 모든 것을)
    if row['Age']==-1:
        train.loc[index,'Age']=medians[row['Name']]
    
for index,row in test.iterrows():
    if row['Age']==-1:
        test.loc[index,'Age']=medians[row['Name']]

train.head()



