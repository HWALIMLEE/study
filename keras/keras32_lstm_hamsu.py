#함수형 모델로 리뉴얼 하시오.
from numpy import array
from keras.models import Model
from keras.layers import Dense,LSTM,Input


x=array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) #1자리가 10개>>10자리가 3개/ weight 1자리에 맞춰짐
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict=array([50,60,70])
print("x.shape:",x.shape) 
x=x.reshape(x.shape[0],x.shape[1],1) 

print("x:",x.shape)
print("x:",x)


#2. 모델구성
# model.add(LSTM(10,activation='relu',input_shape=(3,1)))
input1=Input(shape=(3,1))
dense1_1=LSTM(10,name='A1')(input1)
dense1_2=Dense(10,name='A2')(dense1_1)
dense1_3=Dense(10,name='A3')(dense1_2)
dense1_4=Dense(15,name='A4')(dense1_3)

output1=Dense(20,name='o1')(dense1_4)
output1_2=Dense(10,name='o2')(output1)
output1_3=Dense(10,name='o3')(output1_2)
output1_4=Dense(1,name='o4')(output1_3)

model=Model(input=input1, output=output1_4)
model.summary() 
# 행을 무시하는 이유는 훈련할때는 몇번했는지가 중요하지 않기 때문이다. 즉 데이터의 개수가 몇개인지는 중요하지 않다. 명시할 필요가 없다. 

#3. 실행
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
model.fit(x,y,epochs=10,batch_size=1)

#그러나 예측을 할 때는 데이터의 개수가 주어지고 그것의 형태를 맞춰주어야 한다. 
#(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x_predict=x_predict.reshape(1,3,1)
print(x_predict)

y_predict=model.predict(x_predict)
print(y_predict)
##정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정



