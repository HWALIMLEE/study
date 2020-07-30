from keras.datasets import cifar100
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout
from keras.models import Model, Input
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train=x_train.reshape(50000,32,32,3).astype('float32')/255
x_test=x_test.reshape(10000,32,32,3).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

input1=Input(shape=(32,32,3))
dense1=Conv2D(10,(2,2),activation='relu')(input1)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense3=Dropout(0.2)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense3=Dropout(0.2)
dense3=Conv2D(50,(2,2),activation='relu')(dense2)
dense3=Conv2D(50,(2,2),activation='relu')(dense2)
dense3=Dropout(0.2)
dense3=Conv2D(50,(2,2),activation='relu')(dense2)
dense3=Dropout(0.2)
dense3=Conv2D(30,(2,2),activation='relu')(dense2)
dense3=Dropout(0.2)
dense3=Conv2D(20,(2,2),activation='relu')(dense2)
dense4=Conv2D(10,(3,3),activation='relu')(dense3)
dense5=Dropout(0.2)(dense4)

output1=Conv2D(10,(2,2),activation='relu')(dense5)
output2=Conv2D(10,(2,2),activation='relu')(output1)
output3=MaxPooling2D(pool_size=2)(output2)
output4=Flatten()(output3)
output5=Dense(100,activation="softmax")(output4)

model=Model(input=input1,output=output5)

model.summary()

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

loss_acc=model.evaluate(x_test,y_test, batch_size=50)

#1.
plt.subplot(2,1,1) 
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #compile시킨 loss값
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #fit시킨 값
plt.grid() #모눈종이 처럼 보여주겠다
# plt.legend(['loss','val_loss']) #범례(선에 대한 색깔과 설명)
plt.legend(loc='upper right') #legend의 위치 (location) 명시 안하면 빈곳에 위치
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epoch")


#2.
plt.subplot(2,1,2) 
 #x값은 자동으로 epoch가 될 것
plt.plot(hist.history['acc'],'b-') #metrix값 #compile 시킨 acc값
plt.plot(hist.history['val_acc'],'p-') #fit시킨 값


plt.grid() #모눈종이 처럼 보여주겠다
plt.legend(['acc','val_acc']) #범례(선에 대한 색깔과 설명)
plt.title("acc")
plt.ylabel("acc")
plt.xlabel("epoch")

plt.show() #넣지 않으면 출력이 안된다. 

y_predict=model.predict(x_test)
print(np.argmax(y_predict,axis=1))