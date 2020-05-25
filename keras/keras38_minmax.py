# minmax-min=0, max=1
# standard scaler
import numpy as np
from numpy import array
from keras.models import Model
from keras.layers import Dense,LSTM,Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler


x=array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[2000,3000,4000],[3000,4000,5000],[4000,5000,6000],[100,200,300]]) #오른쪽으로 치우쳐져 있음---> "X만 전처리"
y=array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400]) #y는 전처리 할 필요 없다. 
x_predict=array([55,65,75]) #현재 shape는 (3,)
x_predict=x_predict.reshape(1,3) #reshape먼저 해주어야 함


# MinMaxScaler
# fit하고 transform
# scaler=MinMaxScaler()
# scaler.fit(x) #전처리에서 fit은 실행
# x=scaler.transform(x) 
# print(x)

scalar=StandardScaler()
scalar.fit(x)
x=scalar.transform(x)
# print(x)
# print(np.mean(x))
# print(np.std(x))
print("x.shape:",x.shape) 
x=x.reshape(x.shape[0],x.shape[1],1) 
print("x:",x.shape)
# scaler.fit(x_predict)--->쓰는 것 아님, 이미 x에 훈련시켜준 것(주의!!)
x_predict=scalar.transform(x_predict)
print(x_predict)


#2. 모델구성
# model.add(LSTM(10,activation='relu',input_shape=(3,1)))
input1=Input(shape=(3,1)) 
dense1_1=LSTM(1000,name='A1',return_sequences=True)(input1) #Dense layer는 (행,열)만 입력받는 2차원/return_Sequence=차원 유지--->LSTM과 연결가능해짐/LStM의 output은 2차원 따라서 바로 Dense써도 문제가 없었다. 
dense1_2=LSTM(1000,name='A2')(dense1_1) #원래 있던 차원 형식으로 전달(return_sequences=True)/LSTM은 입력은 3차원, 출력은 2차원
dense1_3=Dense(50,name='A3')(dense1_2)
dense1_4=Dense(1000,name='A4')(dense1_3)
dense1_5=Dense(1000,name='A5')(dense1_4)
dense1_6=Dense(1000,name='A6')(dense1_5)
dense1_7=Dense(100,name='A7')(dense1_6)

output1=Dense(100,name='o1')(dense1_7)
output1_2=Dense(500,name='o2')(output1)
output1_3=Dense(500,name='o3')(output1_2)
output1_4=Dense(300,name='o4')(output1_3)
output1_5=Dense(200,name='o5')(output1_4)
output1_6=Dense(100,name='o6')(output1_5)
output1_7=Dense(1,name='o7')(output1_6)

model=Model(input=input1, output=output1_7)
model.summary() 
#LSTM이 10인경우: param[2]=840///1*10--->bias, 10*10--->역전파, 10(현재)*10(이전)---->input///4*(10+1+10)==4*(input+bias+output)
#ex) LSTM이 11인 경우: param[2]=968///1*11--->bias, 11*11--->역전파, 11(현재)*10(이전)--->input
#LSTM(10)의 10은 output노드의 개수, 원래 그 다음 노드의 개수, 그 다음 시작의 feature와 같다.//증폭된 feature개수는 output노드와 같다. 


#3. 실행
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=1000, mode='aut')
model.fit(x,y,epochs=10000000,batch_size=20,callbacks=[early_stopping])

#그러나 예측을 할 때는 데이터의 개수가 주어지고 그것의 형태를 맞춰주어야 한다. 
#(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x_predict=x_predict.reshape(1,3,1)
print(x_predict)

y_predict=model.predict(x_predict)
print(y_predict)
##정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정

