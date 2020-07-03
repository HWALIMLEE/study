from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout
from keras.models import Model, Input, Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.reshape(50000,32,32,3).astype('float32')/255 #데이터 전처리
x_test=x_test.reshape(10000,32,32,3).astype('float32')/255

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

#sequential형으로 바꾸기
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding="same",activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(32,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))

model.add(Conv2D(64, kernel_size=3, padding="same",activation='relu'))
model.add(Conv2D(64,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))

model.add(Conv2D(128, kernel_size=3, padding="same",activation='relu'))
model.add(Conv2D(128,kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))
# stride default값??

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

# 3. 훈련
# OneHotEncoding쓰지 않고 그냥 하고 싶을 때 loss='sparse_categorical_crossentropy'쓰면 된다. 
# 개인적인 취향
# 통상적으로 OneHotEncoding하고 있어야 함
model.compile(optimizer=Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['acc'])  #0.0004
hist = model.fit(x_train,y_train,
            epochs=20, batch_size=32, verbose=1,
            validation_split=0.3)

"""
#3. 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# 여기서의 loss값과 acc값이 loss,acc로 할당되는 것

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
tb_hist=TensorBoard(log_dir='graph',histogram_freq=0,write_graph=True, write_images=True)


#경로 설정
modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5' #경로/   .hdf5(확장자명)  #d=decimal(정수), float(실수)/2자리 숫자의 정수, 소수 4째자리까지(epoch에 대한 loss값을 파일명으로 확인 가능)
#checkpoint
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True) #filepath-경로(저장이 됨) save_best_only=True(좋은 것만 저장)
hist=model.fit(x_train,y_train,epochs=50,batch_size=50,validation_split=0.1,callbacks=[early_stopping,checkpoint,tb_hist])

plt.figure(figsize=(10,6))
"""
loss = model.evaluate(x_test, y_test)


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('acc 는 ', acc)
# print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# 과적합이 생김
# 훈련 acc: 0.99
# 실제 : loss, acc 는  [2.405737777137756, 0.7190999984741211]
# 과적합 처리방법
# 1. 훈련데이터 늘린다. 
# 2. feature 늘린다.
# 3. 정규화 한다. 