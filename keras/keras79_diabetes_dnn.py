import numpy as np
from sklearn.datasets import load_diabetes
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#1. 데이터
"""
data : x값
target : y값
"""

dataset=load_diabetes()
# print("dataset:",dataset)

x=dataset.data
y=dataset.target
# print("x:",x)
# print("y:",y)
print("x.shape:",x.shape)
print("y.shape:",y.shape)


x=StandardScaler().fit_transform(x)

pca=PCA(n_components=6) #주성분 개수
trans_x=pca.fit_transform(x)

print("trans_x:",trans_x)
print("trans_x.shape:",trans_x.shape)


x_train,x_test,y_train,y_test=train_test_split(trans_x,y,train_size=0.8)
print("x_train:",x_train)
print("x_test:",x_test)

x_train=x_train.reshape(353,6)
x_test=x_test.reshape(89,6)
# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)


print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)
print("y_train:",y_train)



model=Sequential()
model.add(Dense(10,input_shape=(6,),activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='relu'))

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
model.fit(x_train,y_train,epochs=100,batch_size=1,callbacks=[early_stopping])

loss,acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)

print("y_predict:",y_predict)

from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
#RMSE는 함수명
print("RMSE:",RMSE(y_test,y_predict))
#RMSE는 가장 많이 쓰는 지표 중 하나

#R2구하기
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

"""
loss: 5852.006
acc: 5852.005
"""