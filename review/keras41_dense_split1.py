import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

#1.데이터
a=np.array(range(1,11))
size=5

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset=split_x(a,size)
print(dataset)

x=dataset[:,0:4]
y=dataset[:,4]

#2.모델
model=Sequential()
model.add(Dense(10,input_shape=(4,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#3.훈련
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
model.fit(x,y,epochs=100,batch_size=1,callbacks=[early_stopping])

#4.평가
loss,acc=model.evaluate(x,y,batch_size=1)
print("loss:",loss)
print("acc:",acc)
y_predict=model.predict(x)
print("y_predict:",y_predict)
