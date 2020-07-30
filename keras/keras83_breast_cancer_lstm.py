import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

dataset=load_breast_cancer()

x=dataset.data
y=dataset.target

print("x.shape:",x.shape)
print("y.shape:",y.shape)

x=StandardScaler().fit_transform(x)

pca=PCA(n_components=6) #주성분 개수
trans_x=pca.fit_transform(x)

print("trans_x:",trans_x)
print("trans_x.shape:",trans_x.shape)


x_train,x_test,y_train,y_test=train_test_split(trans_x,y,train_size=0.8)
print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("y_train:",y_train)
print("y_test:",y_test)

x_train=x_train.reshape(455,2,3)
x_test=x_test.reshape(114,2,3)

model=Sequential()
model.add(LSTM(10,input_shape=(2,3),activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.summary()

#3. 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
model.fit(x_train,y_train,callbacks=[early_stopping],epochs=10,batch_size=1)

loss,acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict:", np.argmax(y_predict,axis=1))

from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    np.sqrt(mse(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

"""
loss: 0.05
acc: 0.97
"""