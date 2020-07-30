import numpy as np
from keras.models import Input,Model
from keras.layers import Dense, LSTM
from keras.datasets import cifar10
from keras.utils import np_utils

#1. 데이터
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.reshape(50000,32,96).astype('float32')/255
x_test=x_test.reshape(10000,32,96).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#2. 모델 구성
input1=Input(shape=(32,96))
dense1=LSTM(20,activation='relu',return_sequences=True)(input1)
dense2=LSTM(15,activation='relu')(dense1)
dense3=Dense(30,activation='relu')(dense2)
dense4=Dense(20,activation='relu')(dense3)

output1=Dense(20,activation='relu')(dense4)
output2=Dense(30,activation='relu')(output1)
output3=Dense(10,activation='softmax')(output2)

model=Model(input=input1, output=output3)

model.summary()

#3. 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

#4. 평가, 예측
model.fit(x_train,y_train, epochs=10, batch_size=100)
loss,acc=model.evaluate(x_test,y_test,batch_size=100)
y_predict=model.predict(x_test)
print(np.argmax(y_predict,axis=1))