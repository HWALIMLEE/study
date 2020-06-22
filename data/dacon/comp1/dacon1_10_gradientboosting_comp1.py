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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

x=train[0:,0:71]
y=train[0:,71:]
print("x.shape:",x.shape) # (10000, 71)
print("y.shape:",y.shape) # (10000, 4)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1, random_state=60)

scaler=StandardScaler()
scaler.fit(y_train)
y_train_scaled=scaler.transform(y_train)

scaler1=StandardScaler()
scaler1.fit(y_test)
y_test_scaled=scaler1.transform(y_test)


print("x_train.shape:",x_train.shape) #(9000,71)
print("x_test.shape:",x_test.shape)  #(1000,71)

print("y_train:",y_train.shape) #(9000,4)
print("y_test.shape:",y_test.shape) #(1000,4)


pca=PCA(n_components=1, whiten=True,random_state=60).fit(y_train)
y_train_pca=pca.transform(y_train)
y_test_pca=pca.transform(y_test)

print(y_train_pca.shape) #(9000,1)
print(y_test_pca.shape)  #(1000,1)


# parameters={
#     'n_estimators': [10,50,100],
#     'min_samples_leaf':[1,2,4,8,16],
#     'min_samples_split':[1,2,4,8,16]
#     }

# warnings.simplefilter(action='ignore', category=FutureWarning)

#kfold
# kfold=KFold(n_splits=5,shuffle=True)


# 모델구성
model=GradientBoostingRegressor(random_state=0,learning_rate=0.1)

model.fit(x_train,y_train_pca)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test_pca,y_predict)
print("mae:",mae)

print(model.feature_importances_)

print("y_predict:",y_predict)


#평가, 예측
test=test[0:,0:71]

result=model.predict(test)
result=result.reshape(10000,1) #다시 이차원으로 변경
result_transform=pca.inverse_transform(result)

print("result.shape:",result_transform.shape) #(10000,4)로 다시 변경됨
"""
a = np.arange(10000,20000)
#np.arange--수열 만들때
submission = result_transform
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission1_10.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )



"""