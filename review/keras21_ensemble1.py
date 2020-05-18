#1. 데이터 구성
import numpy as np
from numpy import shape
x1=np.array([range(1,101),range(311,411),range(100)])
x2=np.array([range(501,601),range(111,211),range(100)])
y1=np.array([range(301,401),range(511,611),range(101,201)])
y2=np.array([range(101,201),range(411,511),range(811,911)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

#2.분리
from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test,x2_train,x2_test,y2_train,y2_test=train_test_split(x1,y1,x2,y2,shuffle=False,test_size=0.2)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1=Input(shape=(3,)) #질문하기
dense1_1=Dense(30,activation='relu',name='A1')(input1)
dense1_2=Dense(50,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(100,activation='relu',name='A3')(dense1_2)


input2=Input(shape=(3,))
dense2_1=Dense(30,activation='relu',name='B1')(input2)
dense2_2=Dense(50,activation='relu',name='B2')(dense2_1)
dense2_3=Dense(100,activation='relu',name='B3')(dense2_2)

from keras.layers.merge import concatenate
merge1=concatenate([dense1_3,dense2_3],name='merge1')

middle=Dense(30,name='m1')(merge1)
middle1=Dense(50,name='m2')(middle)
middle2=Dense(100,name='m3')(middle1)

output1=Dense(30,name='o1')(middle1)
output1_2=Dense(50,name='o1_2')(output1)
output1_3=Dense(3,name='o1_3')(output1_2)

output2=Dense(30,name='o2')(middle)
output2_2=Dense(50,name='o2_2')(output2)
output2_3=Dense(3,name='o2_3')(output2_2)

model=Model(inputs=[input1,input2],outputs=[output1_3,output2_3])

model.summary()

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=100,batch_size=1,validation_split=0.2)

#4. 평가
loss,o1_3_loss,o2_3_loss,o1_3_mse,o2_3_mse=model.evaluate([x1_test,x2_test],[y1_test,y2_test],batch_size=1)
print("loss:",loss)
print("o1_3_loss:",o1_3_loss)
print("o1_3_mse:",o1_3_mse)
print("o2_3_loss:",o2_3_loss)
print("o2_3_mse:",o2_3_mse)

#5. 예측
y1_predict,y2_predict=model.predict([x1_test,x2_test])

#RMSE
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_predict, y_test):
    return np.sqrt(mse(y_predict, y_test))

RMSE1=RMSE(y1_predict,y1_test)
RMSE2=RMSE(y2_predict,y2_test)

print("RMSE:",(RMSE1+RMSE2)/2)

#R2
from sklearn.metrics import r2_score
r2_1=r2_score(y1_predict, y1_test)
r2_2=r2_score(y2_predict, y2_test)

print("r2:",(r2_1+r2_2)/2)

