import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)

x_train=x_train.reshape(60000,784).astype('float32')/255 
x_test=x_test.reshape(10000,784).astype('float32')/255
print(x_train.shape)

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
model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax')) # 마지막 dense모델에 다중분류는 activation='softmax'가 들어가야 한다. 위에 conv2D에 들어가는 것 아님

model.summary()

# 3. 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
# 분류 모델에서는 accuracy
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실함수는 평균제곱오차 크로스 엔트로피 함수 사용--->다중분류
# 이중분류일떄는 binary_crossentropy
model.fit(x_train,y_train,epochs=50,batch_size=50,callbacks=[early_stopping])

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test,batch_size=50)  #loss, metrics에 집어넣은 값
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict.shape:",y_predict.shape)
print("y_predict:",y_predict)

print(np.argmax(y_predict,axis=1))

#최종값: 0.986
