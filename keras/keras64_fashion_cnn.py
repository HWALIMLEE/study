#Sequential()형으로 완성
#과제2

#하단에 주석으로 acc와 loss결과 명시하시오.

#1. 데이터
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

x_train=x_train.reshape(60000,28,28,1).astype('float32')/255
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

model=Sequential()
model.add(Conv2D(10,(2,2),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(30,(2,2),activation='relu',padding="same"))
model.add(Conv2D(15,(2,2),activation='relu',padding="same"))
model.add(Conv2D(10,(3,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())

model.add(Dense(10,activation='softmax'))

model.summary()

#3.훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(x_train,y_train,epochs=10,batch_size=100)

loss,acc=model.evaluate(x_test,y_test,batch_size=100)

y_predict=model.predict(x_test)

print("y_predict:",np.argmax(y_predict,axis=1))
