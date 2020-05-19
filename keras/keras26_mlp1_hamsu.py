#keras14_mlp를 sequential에서 함수형으로 변경
#1. 데이터(x, y값 준비)
import numpy as np
x=np.array([range(1,101),range(311,411),range(100)]) #--->x=np.transpose(x)로 바꾸자
y=np.array([range(101,201),range(711,811),range(100)])
#리스트-다 모아져 있는 것, [ ]쓰지 않으면 출력이 안 된다. 

x=np.transpose(x)
y=np.transpose(y)

# print(x)
# print(x.shape)
# print(y.shape)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,train_size=0.8)
# print("x_train:",x_train)
# print("x_test:",x_test)


#2. 모델구성 #transfer learning
from keras.models import Sequential,Model
from keras.layers import Dense, Input

input1=Input(shape=(3,))
dense1_1=Dense(10,name='A1')(input1)
dense1_2=Dense(5,name='A2')(dense1_1)
dense1_3=Dense(8,name='A3')(dense1_2)

output1=Dense(40,name='o1')(dense1_3)
output1_2=Dense(30,name='o2')(output1)
output1_3=Dense(50,name='o3')(output1_2)
output1_4=Dense(3,name='o4')(output1_3)

model=Model(input=input1,output=output1_4)
model.summary()

#3.훈련-기계
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x_train, y_train, validation_split=0.2,epochs=100, batch_size=1,callbacks=[early_stopping])
# print("x_train:",x_train)
# print("x_test:",x_test)
# print("x_train_len:",len(x_train))
# print("x_test_len:",len(x_test))

 #validation값 fit에 적용
# metrics에 뜨는 것은 loss, mse, val_loss, val_mse
# val_loss가 loss보다 통상적으로 더 낮다. 
# batch_size가 낮다고 해서 꼭 좋은 loss값이 나오는 것은 아니다. #계속 시행했을 때 acc=1.0나오면 좋은 값 #훈련train


# 4.평가
loss,mse=model.evaluate(x_test,y_test,batch_size=1) #model.evaluate 기본적으로 compile에서 설정한 loss, metrics반환하는 함수 #evaluate는 test값 평가 #그런데 같은 데이터 값으로 다시 평가했음 #과적합
print("loss:",loss)
print("mse:",mse)

#5.예측
#y_pred=model.predict(x_pred)
#print("y_predict:",y_pred)
#훈련데이터와 평가용 데이터는 같은 데이터 쓰게 되면 안된다. 과적합

y_predict=model.predict(x_test) #model에서 예측하는 것이므로 x_test넣었을 때 함수에 해당하는 y값 출력됨
print("y_predict:",y_predict)

#RMSE구하기
from sklearn.metrics import mean_squared_error as mse

#함수는 재사용
#y_test가 원래 값 mse=시그마(y-yhat)^2/n
#y_predict는 x_test로 예측

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
