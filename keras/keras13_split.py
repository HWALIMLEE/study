#1. 데이터(x, y값 준비)
import numpy as np

x=np.array(range(1,101)) #시작지점은 1, 뒤에서 -1빼기
y=np.array(range(101,201)) #weight 값은 1, bias값은 100

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,train_size=0.5,test_size=0.4)
#train_size와 test_size합이 1이 넘어가면 오류가 생김(train_size=1-test_size)
#train_size와 test_size합이 1이 안된다면 어떻게 될까?
#>>비율 정한만큼만 나와서 돌아간다. 
#shuffle은 random_state에 우선한다. 즉, shuffle=False를 하게 되면 random_state에 값을 넣어도 섞이지 않는다. 


# x_train=x[:60] #index의 0은 여기서 1, 60-1=59번째 인덱스=60 #column이 하나이기 때문에 input_dim=1
# x_val=x[60:80] #index의 60은 61, 80-1=79번째 인덱스=80
# x_test=x[80:] #index의 80은 81

# y_train=x[:60]
# y_val=x[60:80]
# y_test=x[80:]


#2. 모델구성 #transfer learning
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(50,input_dim=1)) #x,y한덩어리(input_dim=1)
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(100))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(1000))
# model.add(Dense(5))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(50))
# model.add(Dense(150))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) 여기서부터 실행이 안 됨 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(1))

#3.훈련-기계
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train, y_train, validation_split=0.2,epochs=100, batch_size=1)
print("x_train:",x_train)
print("x_test:",x_test)
print("x_train_len:",len(x_train))
print("x_test_len:",len(x_test))

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
