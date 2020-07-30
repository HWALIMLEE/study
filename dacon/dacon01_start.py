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

x=train[0:,0:71]
y=train[0:,71:]
print("x.shape:",x.shape) # (10000, 71)
print("y.shape:",y.shape) # (10000, 4)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=60)


print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)

x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
test=StandardScaler().fit_transform(test)

print("x_train",x_train)
print("x_test",x_test)

pca=PCA(n_components=10)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
test=pca.fit_transform(test)

x_train=x_train.reshape(8000,10)
x_test=x_test.reshape(2000,10)
test=test.reshape(10000,10)
print(x_train.shape)
print(x_test.shape)

kfold=KFold(n_splits=5,shuffle=True)

# 2. 모델 구성
def create_model():
    model=Sequential()
    model.add(Dense(256,input_shape=(10,),activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(4))
    model.compile(loss='mae',optimizer='adam',metrics=['mae'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error as mae
model=KerasRegressor(build_fn=create_model,epochs=100,verbose=1,batch_size=3)

#4. 평가, 예측
model.fit(x_train,y_train,epochs=100,batch_size=3)

results=cross_val_score(model,x_train,y_train,cv=kfold,n_jobs=1)

print(test.shape)
y_predict=model.predict(x_test)
print(y_predict.shape)

mae_result=mae(y_predict,y_test) #훈련이 얼마나 잘 되었는지 평가

score=model.score(x_test,y_test)
print("r2:",score)
print("result:",results)
print("mae:",mae_result)


"""
mae: 2.58
r2: -2.58
"""