#keras15_mlp를 sequential에서 함수형으로 변경#1. 데이터(x, y값 준비)
import numpy as np

x=np.array([range(1,101),range(311,411),range(100)]) #--->x=np.transpose(x)로 바꾸자
y=np.array(range(711,811))
#리스트-다 모아져 있는 것, [ ]쓰지 않으면 출력이 안 된다. 
x=np.transpose(x)
y=np.transpose(y)
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,test_size=0.2)
# print("x_train:",x_train)
# print("x_test:",x_test)
# print("x_val:",x_val)


#2. 모델구성 #transfer learning
from keras.models import Sequential,Model
from keras.layers import Dense,Input

input1=Input(shape=(3,))
dense1_1=Dense(10,name='A1')(input1)
dense1_2=Dense(50,name='A2')(dense1_1)
dense1_3=Dense(40,name='A3')(dense1_2)

output1=Dense(40,name='o1')(dense1_3)
output1_2=Dense(50,name='o2')(output1)
output1_3=Dense(1,name='o3')(output1_2)

model=Model(input=input1, output=output1_3)
model.summary()


#3.훈련-기계
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=5,mode='aut')
model.fit(x_train, y_train, validation_split=0.2,epochs=100, batch_size=1,callbacks=[early_stopping])
# print("x_train:",x_train)
# print("x_test:",x_test)
# print("x_train_len:",len(x_train))
# print("x_test_len:",len(x_test))


# 4.평가
loss,mse=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("mse:",mse)

#5.예측


y_predict=model.predict(x_test) 
print("y_predict:",y_predict)

#RMSE구하기
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
#RMSE는 함수명
print("RMSE:",RMSE(y_test,y_predict))
#RMSE는 가장 많이 쓰는 지표 중 하나

#R2구하기
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

#즉, RMSE는 낮게 R2는 높게
