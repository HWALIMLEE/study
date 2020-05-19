#keras16_mlp를 sequential에서 함수형으로 변경
#earlyStopping적용
#1. 데이터(x, y값 준비)
import numpy as np

x=np.array(range(1,101))#--->x=np.transpose(x)로 바꾸자
y=np.array([range(101,201),range(711,811),range(100)])


x=np.transpose(x)
y=np.transpose(y)
# print(x)
# print(x.shape)
# print(y.shape)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,test_size=0.2)


#2. 모델구성 #transfer learning
from keras.models import Sequential,Model
from keras.layers import Dense,Input

input1=Input(shape=(1,))
dense1_1=Dense(50,name='A1')(input1)
dense1_2=Dense(100,name='A2')(dense1_1)
dense1_3=Dense(30,name='A3')(dense1_2)

output1=Dense(40,name='o1')(dense1_2)
output1_2=Dense(30,name='o2')(output1)
output1_3=Dense(3,name='o3')(output1_2)

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
loss,mse=model.evaluate(x_test,y_test,batch_size=1) #model.evaluate 기본적으로 compile에서 설정한 loss, metrics반환하는 함수 #evaluate는 test값 평가 #그런데 같은 데이터 값으로 다시 평가했음 #과적합
print("loss:",loss)
print("mse:",mse)


y_predict=model.predict(x_test) #model에서 예측하는 것이므로 x_test넣었을 때 함수에 해당하는 y값 출력됨
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
