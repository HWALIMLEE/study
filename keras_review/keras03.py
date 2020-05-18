import numpy as np
x=np.array([range(1,101),range(301,401),range(100)])
y=np.array(range(100))

x=np.transpose(x)
y=np.transpose(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5,random_state=60)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(100,input_dim=3, activation='relu'))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(70))

model.add(Dense(50))
model.add(Dense(110))
model.add(Dense(15))
model.add(Dense(150))
model.add(Dense(1500))


model.add(Dense(1,activation='relu'))

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit(x_train,y_train,epochs=155,batch_size=5)

loss,mse=model.evaluate(x_test,y_test,batch_size=5)

print("loss:",loss)
print("mse:",mse)


y_predict=model.predict(x_test)
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
print("rmse:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
print("r2:",r2_score(y_test,y_predict))
