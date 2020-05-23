import numpy as np
x1=np.array([range(1,101),range(311,411),range(100)])
y1=np.array([range(301,401),range(511,611),range(101,201)])
y2=np.array([range(501,601),range(111,211),range(100)])

x1=np.transpose(x1)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,y1,y2,random_state=60,test_size=0.2)

from keras.models import Model
from keras.layers import Input,Dense

input1=Input(shape=(3,))
dense1_1=Dense(50,name='A1')(input1)
dense1_2=Dense(40,name='A2')(dense1_1)
dense1_3=Dense(30,name='A3')(dense1_2)
dense1_4=Dense(20,name='A4')(dense1_3)

output1=Dense(4,name='o1')(dense1_3)
output1_2=Dense(5,name='o1_2')(output1)
output1_3=Dense(3,name='o1_3')(output1_2)

output2=Dense(4,name='o2')(dense1_3)
output2_2=Dense(5,name='o2_2')(output2)
output2_3=Dense(3,name='o2_3')(output2_2)

model=Model(input=input1,outputs=[output1_3,output2_3])

model.summary()

#훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=5,mode='aut')
model.fit(x1_train,[y1_train,y2_train],epochs=100,batch_size=1,callbacks=[early_stopping])

loss=model.evaluate(x1_test,[y1_test,y2_test],batch_size=1)

#RMSE
y1_predict,y2_predict=model.predict(x1_test)
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_predict,y_test):
   return np.sqrt(mse(y_predict,y_test))
RMSE1=RMSE(y1_predict,y1_test)
RMSE2=RMSE(y2_predict,y2_test)
print("RMSE:",(RMSE1+RMSE2)/2)

#R2-squared
from sklearn.metrics import r2_score
R1=r2_score(y1_predict,y1_test)
R2=r2_score(y2_predict,y2_test)
print("R2:",(R1+R2)/2)
