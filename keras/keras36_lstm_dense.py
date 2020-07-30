from numpy import array
from keras.models import Model
from keras.layers import Dense,LSTM,Input


x=array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) #1자리가 10개>>10자리가 3개/ weight 1자리에 맞춰짐
y=array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict=array([55,65,75])
print("x.shape:",x.shape)

print("x_predict.shape:",x_predict.shape)

#LSTM---->Dense로 바꾸기(Dense는 LSTM보다 자원이 적게 든다.)
#2. 모델구성
# model.add(LSTM(10,activation='relu',input_shape=(3,1)))
input1=Input(shape=(3,)) 
dense1_1=Dense(10,name='A1')(input1) 
dense1_2=Dense(12,name='A2')(dense1_1) 
dense1_3=Dense(5,name='A3')(dense1_2)
dense1_4=Dense(1,name='A4')(dense1_3)

output1=Dense(20,name='o1')(dense1_4)
output1_2=Dense(10,name='o2')(output1)
output1_3=Dense(10,name='o3')(output1_2)
output1_4=Dense(15,name='o4')(output1_3)
output1_5=Dense(15,name='o5')(output1_4)
output1_6=Dense(15,name='o6')(output1_5)
output1_7=Dense(1,name='o7')(output1_6)

model=Model(input=input1, output=output1_7)
model.summary() 
#LSTM이 10인경우: param[2]=840///1*10--->bias, 10*10--->역전파, 10(현재)*10(이전)---->input///4*(10+1+10)==4*(input+bias+)
#ex) LSTM이 11인 경우: param[2]=968///1*11--->bias, 11*11--->역전파, 11(현재)*10(이전)--->input
#LSTM(10)의 10은 output노드의 개수, 원래 그 다음 노드의 개수, 그 다음 시작의 feature와 같다.//증폭된 feature개수는 output노드와 같다. 


#3. 실행
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=100, mode='aut')
model.fit(x,y,epochs=1000,batch_size=1,callbacks=[early_stopping])

#그러나 예측을 할 때는 데이터의 개수가 주어지고 그것의 형태를 맞춰주어야 한다. 
#(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x_predict=x_predict.reshape(1,3)#--->와꾸 맞춰주기
y_predict=model.predict(x_predict)
print(y_predict)
##정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정



