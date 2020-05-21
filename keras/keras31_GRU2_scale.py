from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM, GRU
#LSTM은 3차원, DENSE는 2차원

#1. 데이터
#원래 데이터는 그냥 1~7까지였음
#batch_size=1이면 (1,2,3) (2,3,4) (3,4,5) (4,5,6) 
#batch_size=2이면 (1,2,3,2,3,4) (3,4,5,4,5,6) 
#feature=1이면 1,2,3,4,5,6,7 한개씩/feature=2이상은 잘 안나옴

x=array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) #1자리가 10개>>10자리가 3개/ weight 1자리에 맞춰짐
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict=array([50,60,70])
print("x.shape:",x.shape) 
x=x.reshape(x.shape[0],x.shape[1],1) 

'''
***batch_size는 행이다.!!! 꼭 기억
              행,       열,      몇개씩 자르는지(batch_size는 아님)
x의 shape=(batch_size,timesteps,feature)  
timesteps
feature-특성(input_dim)

input_shpae=(timesteps,feature)
input_length=timesteps, input_dim=feature

ex)주식으로 생각, 5일치 주가 데이터는 timesteps에 해당
'''

print("x:",x.shape)
print("x:",x)


#2. 모델구성
model=Sequential()
# model.add(LSTM(10,activation='relu',input_shape=(3,1)))
model.add(GRU(100,input_length=3,input_dim=1)) #input_shape(3,1)==(input_length=3, input_dim=1)/ input_dim=1은 1차원/ batch_size는 덩어리로 자르는 거, feature는 행을 자르는 거 
#100행 3열, batch_size를 10으로 하면 만약 데이터가 100개 있다고 하면 1epoch을 돌 때 10개의 step을 통해 epoch을 돌게 됨
#시계열 input_shape=(3,1) ***행 무시***, LSTM에서 중요한 것: 컬럼의 개수와 몇개씩 잘라서 계산할 것이냐, 행은 중요하지 않다
#여기서부터는 Dense모델
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1)) 
model.summary() 
# 행을 무시하는 이유는 훈련할때는 몇번했는지가 중요하지 않기 때문이다. 즉 데이터의 개수가 몇개인지는 중요하지 않다. 명시할 필요가 없다. 

#3. 실행
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
model.fit(x,y,epochs=500,batch_size=1)

#그러나 예측을 할 때는 데이터의 개수가 주어지고 그것의 형태를 맞춰주어야 한다. 
#(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x_predict=x_predict.reshape(1,3,1)
print(x_predict)

y_predict=model.predict(x_predict)
print(y_predict)
##정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정



