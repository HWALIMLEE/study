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

#2. 모델 구성, 훈련
from keras.models import load_model
model=load_model('./model/test1-39-695558.2384.hdf5')

#3. 훈련, 평가
loss_acc=model.evaluate([x_test_scaled,x2_test_scaled],y2_test,batch_size=1)
print("loss_acc:",loss_acc)

y_predict=model.predict([x_test_scaled,x2_test_scaled])

for i in range(5):
    print('종가:',y2_test[i],'/예측가:',y_predict[i])