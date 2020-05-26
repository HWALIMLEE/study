import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1.데이터
a=np.array(range(1,101))
size=5

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset=split_x(a,size)
print("dataset:",dataset)

x=dataset[:90,0:4]
y=dataset[:90,4]
x=x.reshape(90,4,1)

x_predict=dataset[90:96,0:4]
x_predict=x_predict.reshape(6,4,1)


print("x_predict:",x_predict)


#2.모델
model=Sequential()
model.add(LSTM(10,input_shape=(4,1)))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))
model.summary()

#3.훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=30, mode='aut')
model.fit(x,y,callbacks=[early_stopping],epochs=10000,batch_size=1)

#4.평가, 예측
loss, acc=model.evaluate(x,y,batch_size=1)
print("loss:",loss)
print("acc:",acc)
y_predict=model.predict(x_predict)
print("y_predict:",y_predict)