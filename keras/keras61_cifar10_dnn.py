import numpy as np
from keras.models import Model, Input
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape)
x_train=x_train.reshape(50000,3072).astype('float32')/255
x_test=x_test.reshape(10000,3072).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

input1=Input(shape=(3072,))
dense1=Dense(30,activation='relu')(input1)
dense2=Dense(10,activation='relu')(dense1)
dense3=Dense(20,activation='relu')(dense2)
dense4=Dense(15,activation='relu')(dense3)

output1=Dense(20,activation='relu')(dense4)
output2=Dense(10,activation='relu')(output1)
output3=Dense(10,activation='softmax')(output2)

model=Model(input=input1, output=output3)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=20,batch_size=100)

loss,acc=model.evaluate(x_test,y_test,batch_size=100)

y_predict=model.predict(x_test)

print(np.argmax(y_predict,axis=1))

