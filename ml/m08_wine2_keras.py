import numpy as np
import pandas as pd
import sys
import csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.layers import Dense
from keras. models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


# wine=pd.read_csv('./data/csv/winequality-white.csv',sep=';',header=0,index_col=None)
# np.save('./data/winequality-white.npy',arr=wine)

wine=np.load('./data/winequality-white.npy')
print("wine:",wine)

y=wine[0:,11]
x=wine[0:,0:11]
print("y:",y)
print("x:",x)

y_cate=np_utils.to_categorical(y)
print("y_cate.shape:",y_cate.shape)
print("y_cate:",y_cate)
print("x.shape:",x.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y_cate,random_state=10,train_size=0.8)


model=Sequential()
model.add(Dense(10,input_shape=(11,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=10, batch_size=10)

loss_acc=model.evaluate(x_test,y_test,batch_size=1)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)

print("y_predict:",y_predict)

print("loss_acc;",loss_acc)
