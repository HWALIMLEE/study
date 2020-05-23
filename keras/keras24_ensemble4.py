#verbose - 진행되고 있는 것을 설명해주는 것
#1. 데이터(x, y값 준비)
import numpy as np

x1=np.array([range(1,101),range(301,401)])#--->x=np.transpose(x)로 바꾸자
y1=np.array([range(711,811),range(711,811)]) 
y2=np.array([range(101,201),range(411,511)])

###여기서부터 수정 

x1=np.transpose(x1)
y1=np.transpose(y1)
y2=np.transpose(y2)

print(x1.shape)
print(y1.shape)


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,y1,y2,random_state=60,test_size=0.2) #한번에 분리 가능하다. 
# print("x1_train:",x1_train)
# print("x1_test:",x1_test)
# print("y1_train:",y1_train)
# print("y1_test:",y1_test)
# print("y2_train:",y2_train)
# print("y2_test:",y2_test)
print(x1_train.shape) #(80 , 2)
print(y1_test.shape)  #(20 , 2)


#2. 모델구성 함수형으로 바꿈
# 함수형은 서로 다른 모델들을 엮을 수 있다-앙상블
from keras.models import Model #Model-함수형 모델
from keras.layers import Dense, Input #inputlayer명시

# 모델 두개
# activation은 모든 레이어에 다 넣는다. 
# 활성화 함수 기본값은 step function
# dense1으로 다 똑같이 쓰면 하나의 블럭으로 처리됨


#첫번째 모델(1)
input1=Input(shape=(2,)) # 변수명은 소문자(암묵적약속)
dense1_1=Dense(30,activation='relu',name='A1')(input1) #input명시해주어야 함
dense1_2=Dense(4,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(5,activation='relu',name='A3')(dense1_2)
dense1_4=Dense(50,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(100,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(30,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(227,activation='relu',name='A4')(dense1_3)
dense1_4=Dense(300,activation='relu',name='A4')(dense1_3)

#인풋 변수가 하나이기 떄문에 merge쓸 수 없다. 
# from keras.layes.merge import concatenate
# merge1=concatenate([dense1_4,dense2_4])

# 또 레이어 연결(3)--->middle도 없어도 된다. 
# middle1=Dense(3000,name='m1')(dense1_4)
# middle1=Dense(500,name='m2')(middle1)
# middle1=Dense(700,name='m3')(middle1)

# 엮은 거 다시 풀어준다(output도 3개 나와야하니까) ---분리(4)
# input=middle1(상단 레이어의 이름)
output1=Dense(100,name='o1')(dense1_4)
output1_2=Dense(700,name='o1_2')(output1)
output1_3=Dense(250,name='o1_3')(output1_2) 
output1_4=Dense(250,name='o1_4')(output1_3) 
output1_5=Dense(2,name='o1_5')(output1_4) #마지막 아웃풋 2(shape(100,2)이므로)(100,2)짜리가 2개

output2=Dense(100,name='o2')(dense1_4)
output2_2=Dense(700,name='o2_2')(output2)
output2_3=Dense(250,name='o2_3')(output2_2) 
output2_4=Dense(250,name='o2_4')(output2_3) 
output2_5=Dense(2,name='o2_5')(output2_4) 

#함수형 지정(제일 하단에 명시함)
model=Model(input=input1, outputs=[output1_5,output2_5]) # 범위 명시 #함수형은 마지막에 선언 #두 개 이상은 리스트

model.summary()


##batch_size같이 맞춰주는 게 좋음
#3.훈련-기계
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
model.fit(x1_train,[y1_train,y2_train],validation_split=0.2,epochs=200,batch_size=8,verbose=1) #두 개 이상일떄는 리스트


# 4.평가
# 출력값이 여러개
# evaluate 에 batch_size값 명시 안된 경우 많다. defalult값은 16
loss,o1_5_loss,o2_5_loss,o1_5_mse,o2_5_mse=model.evaluate(x1_test,[y1_test,y2_test],batch_size=5) 
print("loss:",loss)
# print("mse:",mse)


#5.예측
y1_predict,y2_predict=model.predict(x1_test) 
# print("y1_predict:",y1_predict)
# print("y2_predict:",y2_predict)
# print("y3_predict:",y3_predict)
print("y1_predict:",y1_predict)
print("y2_predict:",y2_predict)


#RMSE구하기
from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

RMSE1=RMSE(y1_test,y1_predict)
RMSE2=RMSE(y2_test,y2_predict)
# RMSE3=RMSE(y3_test,y3_predict)
print("RMSE1:",(RMSE1+RMSE2)/2)
# print("RMSE2:",RMSE2)
# print("RMSE3:",RMSE3)
# print("RMSE:",(RMSE1+RMSE2+RMSE3)/3)


#R2구하기
from sklearn.metrics import r2_score
r2_1=r2_score(y1_predict,y1_test)
r2_2=r2_score(y2_predict,y2_test)
# r2_3=r2_score(y3_predict,y3_test)
print("R2:",(r2_1+r2_2)/2)
print("R2_1:",r2_1)
print("R2_2:",r2_2)
# print("R2_3:",r2_3)
#즉, RMSE는 낮게 R2는 높게

