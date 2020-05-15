import numpy as np
x=np.array(range(1,101))
y=np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x,y,random_state=60,test_size=0.3)
x_val,x_test,y_val,y_test=train_test_split(x_train,y_train,random_state=60, train_size=0.5)

print("x_val:",x_val)
print("x_val_len:",len(x_val))

#2. 모델구성
from keras.layers import Dense
from keras.models import Sequential

model=Sequential()
model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1,activation='relu'))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])

#4. 평가 예측
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_data=(x_val,y_val))
loss,mse=model.evaluate(x_test,y_test,batch_size=1)

#5. rmse구하기
y_predict=model.predict(x_test)
from sklearn.metrics import mean_squared_error as mse
def rmse(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
print("rmse:",rmse(y_test,y_predict))

#6. r-squared구하기
from sklearn.metrics import r2_score
print("r2:",r2_score(y_predict,y_test))

