#Sequential()형으로 완성하시오.
#과제4

#하단에 주석으로 acc와 loss결과 명시하시오.
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train=x_train.reshape(60000,28,28).astype('float32')/255
x_test=x_test.reshape(10000,28,28).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#2. 모델구성
model=Sequential()
model.add(LSTM(10,input_shape=(28,28),activation='relu',return_sequences=True))
model.add(LSTM(20,activation='relu',return_sequences=True))
model.add(LSTM(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

#3. 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(patience=5,monitor='acc',mode='max')
model.fit(x_train,y_train,epochs=50, batch_size=50,callbacks=[early_stopping])

loss,acc=model.evaluate(x_test,y_test,batch_size=50)

print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print(np.argmax(y_predict,axis=1))


"""
loss: 0.333
acc: 0.881
"""