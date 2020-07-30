# 모두 함수형으로 만들기
import numpy as np
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.datasets import cifar10
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler

# scaler=MinMaxScaler() --->minmaxscaler 쓰려면 이차원에서 해야하기 때문에 reshape 전에 해야한다.
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.reshape(50000,32,32,3).astype('float32')/255
x_test=x_test.reshape(10000,32,32,3).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("y_train.shape:",y_train.shape)

print("x_train.shape:",x_train.shape)

input1=Input(shape=(32,32,3))
dense1=Conv2D(10,(2,2),activation='relu')(input1)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense3=Conv2D(50,(2,2),activation='relu')(dense2)
dense4=Conv2D(30,(3,3),activation='relu')(dense3)
dense5=Dropout(0.2)(dense4)

output1=Conv2D(50,(2,2),activation='relu')(dense5)
output2=Conv2D(30,(2,2),activation='relu')(output1)
output3=MaxPooling2D(pool_size=2)(output2)
output4=Flatten()(output3)
output5=Dense(10,activation="softmax")(output4)

model=Model(input=input1,output=output5)

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(x_train,y_train,epochs=10, batch_size=100)

loss,acc=model.evaluate(x_test,y_test,batch_size=100)

y_predict=model.predict(x_test)
print("y_predict.shape:",y_predict.shape)

print(np.argmax(y_predict,axis=1))
