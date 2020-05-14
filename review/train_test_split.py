#1. 데이터 생성
import numpy as np
x=np.array(range(1,101))
y=np.array(range(101,201))
print("x:",x)
print("y:",y)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=90,test_size=0.2)
x_val,x_test,y_val,y_test=train_test_split(x_train,y_train,random_state=90,test_size=0.8)


# print("x_train:\n",x_train)
# print("x_test:\n",x_test)
# print("y_train:\n",y_train)
# print("y_test:\n",y_test)

model=Sequential()
model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1,activation='relu'))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])

#4.평가, 예측
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))
loss,mse=model.evaluate(x_test,y_test,batch_size=1)

#5. rmse
y_predict=model.predict(x_test)
from sklearn.metrics import mean_squared_error as mse
def rmse(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
print("rmse:",rmse(y_test,y_predict))

#6. R-squared
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("r2:",r2)

#실수한 부분
# 1. model.compile은 모델만 학습시키는 것이기 때문에 따로 변수가 들어가지 않는다. 
# 2. model.compile에는 validation_data도 들어가지 않는다. 
# 3. validaion_data는 model에 fitting 시킬 때 들어가는 것
# 4. fitting시킬 때는 train으로 훈련시키기
# 5. 평가할 때는 test값으로!
# 6. rmse와 r2값은 sklearn.metrics에 있다. 