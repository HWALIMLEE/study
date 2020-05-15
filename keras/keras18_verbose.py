#verbose - 진행되고 있는 것을 설명해주는 것

#1. 데이터(x, y값 준비)
import numpy as np

#시작지점은 1, 뒤에서 -1빼기 range여러개 나열하면 오류가 나옴--->리스트로 만들면 된다. 
#3행 100열로 나오게 된다. >>>100행 3열로 바꿔야 함
#바꾸려면....?
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
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,train_size=0.5,test_size=0.2)#column채로 잘리게 된다. train=(80,3) test=(20,3) 행의 숫자에 맞춰서 잘림
x_val, x_test, y_val, y_test=train_test_split(x_test,y_test, shuffle=False, test_size=0.5) 
# print("x_train:",x_train)
# print("x_test:",x_test)
# print("x_val:",x_val)


#2. 모델구성 #transfer learning
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(50,input_dim=3)) #x,y한덩어리(input_dim=1)
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #output을 3으로 하면 3개이 데이터를 넣었을 때 3개를 예측하는 것은 이상하다. 3개의 상호작용을 통해 하나를 예측하는 것이 더 설득력 있다.

#3.훈련-기계
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train, y_train, validation_split=0.2,epochs=100, batch_size=1,verbose=3)
# verbose 종류: verbose=0, verbose=1, verbose=2, verbose=3
# verbose=0은 metric이 나오지 않는다. 
# verbose=1,2,3 나에게 보여주는 시간만큼 delay되는 것, verbose=0이 가장 빨리 훈련이 된다. (보여주지 않기 때문에)
# verbose=1,2,3 은 선택해서 사용하면 된다. 


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
