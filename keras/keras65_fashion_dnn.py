#Sequential()형으로 완성
#과제3

#하단에 주석으로 acc와 loss결과 명시하시오.

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


model=Sequential()
model.add(Dense(20,input_shape=(784,)))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
# model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=300, batch_size=150)

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test,batch_size=50)

print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict:",np.argmax(y_predict,axis=1))

"""
loss: 0.41
acc: 0.87
"""