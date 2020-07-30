#머신러닝은 npy로 저장해줄 필요 없다
import numpy as np
import pandas as pd
import sys
import csv


wine=pd.read_csv('./data/csv/winequality-white.csv',sep=';',header=0,index_col=None)
# wine_value=wine.values

print("wine:",wine)
print(wine.shape)
y=wine["quality"]
print("y:",y)
x=wine.iloc[0:,0:11]
print("x:",x)
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,train_size=0.8)

#모델구성

model=RandomForestClassifier()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

# r2=r2_score(y_test,y_predict)
score=model.score(x_test,y_test) #자동반환, acc와 r2값 중
acc=accuracy_score(y_test,y_predict)

print("y_predict:",y_predict)
# print("r2:",r2)                
print("score:",score)         
print("acc:",acc)         #분류모델이므로 acc안나옴