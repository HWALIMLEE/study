#앙상블 모델로 만드시오.(input:output=2:1)
from numpy import array
from keras.models import Model
from keras.layers import Dense,LSTM,Input


x1=array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2=array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]) 
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

x1=x1.reshape(x1.shape[0],x1.shape[1],1) 
x2=x2.reshape(x2.shape[0],x2.shape[1],1)
print("x1.shape:",x1.shape) 
print("x2.shape:",x2.shape)

#2. 모델구성
input1=Input(shape=(3,1)) # 변수명은 소문자(암묵적약속)
dense1_1=LSTM(30,activation='relu',name='A1')(input1) #input명시해주어야 함
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(50,activation='relu',name='A3')(dense1_2)
dense1_4=Dense(20,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(30,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(20,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(10,activation='relu',name='A4')(dense1_3)


input2=Input(shape=(3,1)) # 변수명은 소문자(암묵적약속)
dense2_1=LSTM(30,activation='relu',name='B1')(input2) #input명시해주어야 함
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_3=Dense(50,activation='relu',name='B3')(dense2_2)
dense2_4=Dense(30,activation='relu',name='B4')(dense2_3)
dense2_4=Dense(10,activation='relu',name='B4')(dense2_3)
dense2_4=Dense(30,activation='relu',name='B4')(dense2_3)
dense2_4=Dense(20,activation='relu',name='B4')(dense2_3)

from keras.layers.merge import concatenate
merge1=concatenate([dense1_4,dense2_4],name='merge1')

middle1=Dense(30,name='m1')(merge1)
middle1=Dense(50,name='m2')(middle1)
middle1=Dense(70,name='m3')(middle1)

output1=Dense(10,name='o1')(middle1)
output1_2=Dense(70,name='o1_2')(output1)
output1_3=Dense(25,name='o1_3')(output1_2) 
output1_4=Dense(25,name='o1_4')(output1_3) 
output1_5=Dense(1,name='o1_5')(output1_4)

model=Model(inputs=[input1, input2],output=output1_5)
model.summary()

#3. 실행
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10, mode='aut')
model.fit([x1,x2],y,epochs=100,batch_size=1)

#그러나 예측을 할 때는 데이터의 개수가 주어지고 그것의 형태를 맞춰주어야 한다. 
#(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x1_predict=x1_predict.reshape(1,3,1)
x2_predict=x2_predict.reshape(1,3,1)
# print(x1_predict.shape)
# print(x2_predict.shape)

# print(x1.shape)
# print(x2.shape)

y_predict=model.predict([x1_predict,x2_predict]) #처음 모델이 x 2개를 넣어서 y하나 예측하는 것이었음 따라서 predict도 동일하게!

 #대괄호 써주어야 함 리스트로 만들기
print("y_predict:",y_predict)



