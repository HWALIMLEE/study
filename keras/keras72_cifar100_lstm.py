import numpy as np
from keras.models import Model,Sequential, Input
from keras.layers import LSTM, Dense,Dropout
from keras.datasets import cifar100
from keras.utils import np_utils
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train=x_train.reshape(50000,96,32).astype('float32')/255
x_test=x_train.reshape(10000,96,32).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#2. 모델 구성
input1=Input(shape=(96,3))
dense1_1=LSTM(10,activation='relu')(input1)
dense1_2=Dense(10,activation='relu')(dense1_1)
dense1_3=Dense(10,activation='relu')(dense1_2)
dense1_4=Dense(15,activation='relu')(dense1_3)

output1=Dense(20,activation='relu')(dense1_4)
output1_2=Dense(10,activation='relu')(output1)
output1_3=Dense(10,activation='relu')(output1_2)
output1_4=Dense(100,activation='softmax')(output1_3)

model=Model(input=input1, output=output1_4)
model.summary() 

#3. 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='auto')
modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True)

tb_hist=TensorBoard(log_dir='graph',histogram_freq=0,write_graph=True,write_images=True)
hist=model.fit(x_train,y_train,epochs=300, batch_size=150,validation_split=0.2,callbacks=[early_stopping,checkpoint,tb_hist])
print(hist)
print(hist.history.keys())

loss_acc=model.evalute(x_test,y_test,batch_size=100)

plt.figure(figsize=(10,6))

#1. 
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #compile시킨 값
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #fit시킨 값
plt.grid()
plt.legend(loc='upper right')
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epoch")

#2.
plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.',c='red',label='acc') #compile시킨 값
plt.plot(hist.history['val_acc'],marker='.',c='blue',label='val_acc') #fit시킨 값
plt.grid()
plt.legend(loc='upper right')
plt.title("acc")
plt.ylabel("acc")
plt.xlabel("epoch")

plt.show()

#4. 평가, 예측

y_predict=model.predict(x_test)
print("y_predict:",np.argmax(y_predict,axis=1))