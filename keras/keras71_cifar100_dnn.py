import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential,Model,Input
from keras.utils import np_utils
from keras.datasets import cifar100
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=cifar100.load_data()

x_train=x_train.reshape(50000,3072).astype('float32')/255
x_test=x_test.reshape(10000,3072).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)


input1=Input(shape=(3072,))
dense1=Dense(10,activation='relu')(input1)
dense2=Dense(10,activation='relu')(dense1)
dense3=Dense(20,activation='relu')(dense2)
dense4=Dense(20,activation='relu')(dense3)
dense4=Dropout(0.2)(dense3)
dense5=Dense(30,activation='relu')(dense4)
dense6=Dense(30,activation='relu')(dense5)
dense7=Dense(30,activation='relu')(dense6)
dense8=Dropout(0.2)(dense7)
dense9=Dense(15,activation='relu')(dense8)

output1=Dense(20,activation='relu')(dense4)
output2=Dropout(0.3)(output1)
output3=Dense(10,activation='relu')(output2)
output4=Dense(10,activation='relu')(output3)
output5=Dense(10,activation='relu')(output4)
output6=Dense(10,activation='relu')(output5)
output7=Dense(10,activation='relu')(output6)
output8=Dense(10,activation='relu')(output7)
output9=Dense(100,activation='softmax')(output8)

model=Model(input=input1, output=output9)

model.summary()


#3. 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#loss값, acc값 반환
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
early_stopping=EarlyStopping(monitor='loss',patience=10, mode='aut')

modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5' #epoch은 두자리 정수, val_loss는 4자리 실수

checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True)

tb_hist=TensorBoard(log_dir='graph',histogram_freq=0,write_graph=True,write_images=True)
hist=model.fit(x_train,y_train,epochs=300, batch_size=150,validation_split=0.2,callbacks=[early_stopping,checkpoint,tb_hist])
print(hist)
print(hist.history.keys())

loss_acc=model.evaluate(x_test,y_test,batch_size=150)

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