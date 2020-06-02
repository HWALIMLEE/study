import pandas as pd
import numpy as np
from keras.models import Input,Model
from keras.layers import Dense, LSTM
import random

samsung=np.load('./data/samsung.npy',allow_pickle=True)
hite=np.load('./data/hite.npy',allow_pickle=True)

# print("samsung:",samsung)
# print("hite:",hite)

print("samsung.shape:",samsung.shape)
# print("hite.shape:",hite.shape)

#hite에 6.2 시가 넣어주기
hite=hite[0:,0]
hite=np.append(hite,np.array([39000]),axis=0)
print("hite.shape:",hite.shape)
hite=hite.reshape(509,1)
print(hite)
print("hite.shape:",hite.shape)


# def split_xy5(dataset,time_steps,y_column):
#     x,y=list(),list()
#     for i in range(len(dataset)):
#         x_end_number=i+time_steps
#         y_end_number=x_end_number + y_column

#         if y_end_number>len(dataset):
#             break
#         tmp_x=dataset[i:x_end_number,:]
#         tmp_y=dataset[x_end_number : y_end_number,3]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x),np.array(y)


def split(dataset,time_steps,y_column):
    x,y=list(),list()
    for i in range(len(dataset)):
        x_end_number=i+time_steps
        y_end_number=x_end_number + y_column

        if y_end_number>len(dataset):
            break
        tmp_x=dataset[i:x_end_number,:]
        tmp_y=dataset[x_end_number:y_end_number,:]
        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x),np.array(y)

x,y=split(hite,5,1)
print("x.shape:",x.shape)
y=y.reshape(504,1)
print("y.shape:",y.shape)


x2,y2=split(samsung,5,1)
print("x2.shape:",x2.shape)
y2=y2.reshape(504,1)
print("y2.shape:",y2.shape)


#나누기
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,random_state=1,test_size=0.2)
print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("x2_train.shape:",x2_train.shape)
print("x2_test.shape:",x2_test.shape)


#reshape-->2차원으로 변경
x_train=x_train.reshape(403,5)
x_test=x_test.reshape(101,5)
x2_train=x2_train.reshape(403,5)
x2_test=x2_test.reshape(101,5)

#표준화
from sklearn.preprocessing import StandardScaler
scaler1=StandardScaler()
scaler1.fit(x_train)
x_train_scaled=scaler1.transform(x_train)
x_test_scaled=scaler1.transform(x_test)
print(x_train_scaled[0,:])
print(x_test_scaled[0,:])

scaler2=StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled=scaler2.transform(x2_train)
x2_test_scaled=scaler2.transform(x2_test)
print(x2_train_scaled[0,:])
print(x2_test_scaled[0,:])



#3차원 배열로 변경
x_train_scaled=x_train_scaled.reshape(403,5,1)
x_test_scaled=x_test_scaled.reshape(101,5,1)


x2_train_scaled=x2_train_scaled.reshape(403,5,1)
x2_test_scaled=x2_test_scaled.reshape(101,5,1)


input1=Input(shape=(5,1)) # 변수명은 소문자(암묵적약속)
dense1_1=LSTM(30,activation='relu',name='A1')(input1) #input명시해주어야 함
dense1_2=Dense(20,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(20,activation='relu',name='A3')(dense1_2)
dense1_4=Dense(40,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(50,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(30,activation='relu',name='A4')(dense1_3)

#두번쨰 모델(2)
input2=Input(shape=(5,1)) 
dense2_1=LSTM(30,activation='relu',name='B1')(input2) 
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_3=Dense(30,activation='relu',name='B3')(dense2_2)
dense2_4=Dense(50,activation='relu',name='B4')(dense2_3)
dense2_5=Dense(50,activation='relu',name='B5')(dense2_4)
dense2_5=Dense(50,activation='relu',name='B5')(dense2_4)
dense2_5=Dense(30,activation='relu',name='B5')(dense2_4)

# 엮어주는 기능(첫번째 모델과 두번째 모델)(3) #concatenate-사슬 같이 잇다, 단순병합
from keras.layers.merge import concatenate
merge1=concatenate([dense1_4,dense2_5],name='merge1') #두 개 이상은 항상 리스트('[]')

# 또 레이어 연결(3)
middle1=Dense(30,name='m1')(merge1)
middle1=Dense(50,name='m2')(middle1)
middle1=Dense(100,name='m3')(middle1)
middle1=Dense(50,name='m4')(middle1)
middle1=Dense(20,name='m5')(middle1)

# input=middle1(상단 레이어의 이름)
output1=Dense(10,name='o1')(middle1)
output1_2=Dense(50,name='o1_2')(output1)
output1_2=Dense(50,name='o1_2')(output1)
output1_3=Dense(30,name='o1_3')(output1_2) 
output1_3=Dense(30,name='o1_3')(output1_2) 
output1_4=Dense(30,name='o1_4')(output1_3) 
output1_5=Dense(1,name='o1_5')(output1_4) 


#함수형 지정(제일 하단에 명시함)
model=Model(inputs=[input1,input2], outputs=output1_5) # 범위 명시 #함수형은 마지막에 선언 #두 개 이상은 리스트

model.summary()


#3.훈련-기계
modelpath='./model/test4-{epoch:02d}-{val_loss:.4f}.hdf5'
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
from keras.callbacks import EarlyStopping,ModelCheckpoint
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='min')
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=1)
model.fit([x_train_scaled,x2_train_scaled],y2_train,epochs=200, batch_size=3,callbacks=[early_stopping,checkpoint],validation_split=0.2)

model.save('./model/test4_lstm.h5')


#4. 훈련, 평가
loss_acc=model.evaluate([x_test_scaled,x2_test_scaled],y2_test,batch_size=1)
print("loss_acc:",loss_acc)

y_predict=model.predict([x_test_scaled,x2_test_scaled])



from sklearn.metrics import mean_squared_error as mse
def RMSE(y2_test,y_predict):
    return np.sqrt(mse(y2_test,y_predict))

print("RMSE:",RMSE(y2_test,y_predict))


from sklearn.metrics import r2_score
print("r2:",r2_score(y2_test,y_predict))

# y_predict_1 = scaler.inverse_transform(y_predict)
for i in range(5):
    print('종가:',y2_test[i],'/예측가:',y_predict[i])
