#1. 데이터
import numpy as np
x=np.array([range(1,101),range(311,411),range(100)])
y=np.array([range(101,201),range(711,811),range(100)])
x=np.transpose(x)
y=np.transpose(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=3))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

#4. 평가, 예측
loss,mse=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("mse:",mse)

#5. RMSE
from sklearn.metrics import mean_squared_error as mse
y_predict=model.predict(x_test)
def RMSE(y_predict,y_test):
    return np.sqrt(mse(y_predict,y_test))
print("RMSE:",RMSE(y_predict,y_test))

#6. R2 score
from sklearn.metrics import r2_score
r2=r2_score(y_predict,y_test)
print("R2:",r2)
