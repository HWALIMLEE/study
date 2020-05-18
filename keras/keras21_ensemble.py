#verbose - 진행되고 있는 것을 설명해주는 것
#1. 데이터(x, y값 준비)
import numpy as np

x1=np.array([range(1,101),range(311,411),range(100)]) #--->x=np.transpose(x)로 바꾸자
y1=np.array([range(711,811),range(711,811),range(100)]) #--->x=np.transpose(x)로 바꾸자

x2=np.array([range(101,201),range(411,511),range(100,200)]) #--->x=np.transpose(x)로 바꾸자
y2=np.array([range(501,601),range(711,811),range(100)])
#리스트-다 모아져 있는 것, [ ]쓰지 않으면 출력이 안 된다. 

x1=np.transpose(x1)
y1=np.transpose(y1)

x2=np.transpose(x2)
y2=np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test,x2_train,x2_test,y2_train,y2_test=train_test_split(x1,y1,x2,y2,shuffle=False,test_size=0.2)

#2. 모델구성 함수형으로 바꿈
# 함수형은 서로 다른 모델들을 엮을 수 있다-앙상블
from keras.models import Model #Model-함수형 모델
from keras.layers import Dense, Input #inputlayer명시

# 모델 두개
# activation은 모든 레이어에 다 넣는다. 
# 활성화 함수 기본값은 step function
# dense1으로 다 똑같이 쓰면 하나의 블럭으로 처리됨


#첫번째 모델(1)
input1=Input(shape=(3,)) # 변수명은 소문자(암묵적약속)
dense1_1=Dense(30,activation='relu',name='A1')(input1) #input명시해주어야 함
dense1_2=Dense(400,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(50,activation='relu',name='A3')(dense1_2)
dense1_4=Dense(30,activation='relu',name='A4')(dense1_3)

#두번쨰 모델(2)
input2=Input(shape=(3,)) 
dense2_1=Dense(50,activation='relu',name='B1')(input2) 
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_3=Dense(100,activation='relu',name='B3')(dense2_2)
dense2_4=Dense(30,activation='relu',name='B4')(dense2_3)

# 엮어주는 기능(첫번째 모델과 두번째 모델)(3) #concatenate-사슬 같이 잇다, 단순병합
from keras.layers.merge import concatenate
merge1=concatenate([dense1_4,dense2_4],name='merge1') #두 개 이상은 항상 리스트('[]')

# 또 레이어 연결(3)
middle1=Dense(30,name='m1')(merge1)
middle1=Dense(5,name='m2')(middle1)
middle1=Dense(7,name='m3')(middle1)

# 엮은 거 다시 풀어준다(output도 2개 나와야하니까) ---분리(4)
# input=middle1(상단 레이어의 이름)
output1=Dense(10,name='o1')(middle1)
output1_2=Dense(70,name='o1_2')(output1)
output1_3=Dense(30,name='o1_3')(output1_2) 
output1_4=Dense(30,name='o1_4')(output1_3) 
output1_5=Dense(3,name='o1_5')(output1_4) #마지막 아웃풋 3(shape(100,3)이므로)

#input=middle1(5)
output2=Dense(10,name='o2')(middle1)
output2_2=Dense(70,name='o2_2')(output2)
output2_3=Dense(30,name='o2_3')(output2_2)
output2_4=Dense(30,name='o2_4')(output2_3)
output2_5=Dense(3,name='o2_5')(output2_4) #output의 노드의 개수는 input의 노드의 개수와 동일

#함수형 지정(제일 하단에 명시함)
model=Model(inputs=[input1,input2], outputs=[output1_5, output2_5]) # 범위 명시 #함수형은 마지막에 선언 #두 개 이상은 리스트

model.summary()


##batch_size같이 맞춰주는 게 좋음
#3.훈련-기계
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit([x1_train,x2_train],[y1_train,y2_train],validation_split=0.2,epochs=10,batch_size=1,verbose=1) #두 개 이상일떄는 리스트


# 4.평가
# 출력값이 여러개
loss,o1_5_loss,o2_5_loss,o1_5_mse,o2_5_mse=model.evaluate([x1_test,x2_test],[y1_test,y2_test],batch_size=1) #loss, o1_5_loss, o2_5_loss, o1_5_mse, o2_5_mse output이 2개이므로
print("loss:",loss)
# print("mse:",mse)


#5.예측
y_predict=model.predict([x1_test,x2_test])
y1_predict,y2_predict=model.predict([x1_test,x2_test]) 
print("y1_predict:",y1_predict)
print("y1_predict_len:",len(y1_predict)) #y1_predict의 개수 세어보기(20개)
print("y2_predict:",y2_predict)
print("y_predict:",y_predict)


#RMSE구하기
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

RMSE1=RMSE(y1_test,y1_predict)
RMSE2=RMSE(y2_test,y2_predict)
print("RMSE1:",RMSE1)
print("RMSE2:",RMSE2)
print("RMSE:",(RMSE1+RMSE2)/2)


#R2구하기
from sklearn.metrics import r2_score
r2_1=r2_score(y1_predict,y1_test)
r2_2=r2_score(y2_predict,y2_test)
print("R2_1:",r2_1)
print("R2_2:",r2_2)
print("R2:",(r2_1+r2_2)/2)
#즉, RMSE는 낮게 R2는 높게
