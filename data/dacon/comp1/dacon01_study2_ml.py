import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train=pd.read_csv('./data/dacon/comp1/train.csv',header=0,index_col=0)  #0행이 header, 0열이 index/ header와 index모두 존재
test=pd.read_csv('./data/dacon/comp1/test.csv',header=0, index_col=0)
submission=pd.read_csv('./data/dacon/comp1/sample_submission.csv',header=0,index_col=0)

print("train.shape:",train.shape)           # (10000, 75) # x_train , x_test , y_train , y_test/ 평가도 train으로
print("test.shape:",test.shape)             # (10000, 71) # x_predict가 된다 # y값이 없다
print("submission.shape:",submission.shape) # (10000, 4)  # y_predict가 된다

# test + submission = train
# test는 y값이 없음

#이상치는 알 수 없으나 결측치는 알 수 있다.
print(train.isnull().sum())

train=train.interpolate() #보간법//선형//완벽하진 않으나 평타 85%//컬럼별로 선을 잡아서 빈자리 선에 맞게 그려준다//컬럼별 보간
train=train.fillna(method='bfill')
print(train.isnull().sum())
print("train:",train.head())
print(test.isnull().sum())
test=test.interpolate()
test=test.fillna(method='bfill')
print("test:",test.head())

np.save('./data/comp1_train.npy',arr=train)
np.save('./data/comp1_test.npy',arr=test)

# 1. 데이터
train=np.load('./data/comp1_train.npy')
test=np.load('./data/comp1_test.npy')

from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


x=train[0:,0:71]
y=train[0:,71:]
print("x.shape:",x.shape) # (10000, 71)
print("y.shape:",y.shape) # (10000, 4)
print("test.shape:",test.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=60)


print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)

# 37열부터 70열까지 거의 모든 데이터 값이 0에 가까웠다. 따라서 따로 표준화시켜보기
x_train1=x_train[0:,0:1] 
x_train2=x_train[0:,1:37]
x_train3=x_train[0:,37:71]

print(x_train1.shape)
print(x_train2.shape) 
print(x_train3.shape)

x_train1=StandardScaler().fit_transform(x_train1)
x_train2=StandardScaler().fit_transform(x_train2)
x_train3=StandardScaler().fit_transform(x_train3)
x_train=np.hstack((x_train1,x_train2,x_train3))
print(x_train.shape)

x_test1=x_test[0:,0:1] 
x_test2=x_test[0:,1:37]
x_test3=x_test[0:,37:71]

x_test1=StandardScaler().fit_transform(x_test1)
x_test2=StandardScaler().fit_transform(x_test2)
x_test3=StandardScaler().fit_transform(x_test3)
x_test=np.hstack((x_test1,x_test2,x_test3))

print("x_test.shape",x_test.shape)

pca=PCA(n_components=40, whiten=True,random_state=60).fit(x_train)
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)

model=RandomForestRegressor()

model.fit(x_train_pca,y_train)

y_predict=model.predict(x_test_pca)


test_pca=pca.transform(test)
result=model.predict(test_pca)


from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_predict)
print("mae:",mae)
a = np.arange(10000,20000)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission_study2.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )

