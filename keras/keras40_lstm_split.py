import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1. 데이터
a=np.array(range(1,11))
size=5          #time_steps=4(얼만큼 자를 것이냐)
#1,2,3,4,5//2,3,4,5,6//3,4,5,6,7//4,5,6,7,8//5,6,7,8,9//6,7,8,9,10
#데이터 자르기

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)] #행 구성
        aaa.append([item for item in subset]) #subset에 있는 아이템을 반환
    print(type(aaa))
    return np.array(aaa)
    #return 값이 np.array

dataset=split_x(a,size)
print("================")
print(dataset)

#외우기!!!
x=dataset[:,0:4] # 행은 전체, 열은 0:4
y=dataset[:,4] 
print(x)
print(y)

#reshape먼저 시켜주어야 함
#numpy로
x=x.reshape(6,4,1)
#여기서의 batch_size의 의미는 총 행(자를 것이 총 몇개인지)
#x=np.reshape(x,(6,4,1)) 똑같은 방법
print(x.shape)

# LSTM 모델을 완성하시오.

#2. 모델
model=Sequential()

model.add(LSTM(10,input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(1))
model.summary() 
# 행을 무시하는 이유는 훈련할때는 몇번했는지가 중요하지 않기 때문이다. 즉 데이터의 개수가 몇개인지는 중요하지 않다. 명시할 필요가 없다. 

#3. 훈련
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=100, mode='auto')
#patience는 epoch보다 작아야 한다. 
model.fit(x,y,epochs=10000,batch_size=1,callbacks=[early_stopping])

#4. 평가, 예측
loss,acc=model.evaluate(x,y,batch_size=1) #metrics 꼭 써주어야 함
print("loss:",loss)
print("acc:",acc)
y_predict=model.predict(x)
print("y_predict:",y_predict)