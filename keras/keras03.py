#1. 데이터
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10]) #crtl+c->ctrl+v --->한 라인 카피 shift+delete--->라인 전체 지우기  ctrl+'/'--->한 라인 주석처리

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(5,input_dim=1)) #x,y한덩어리(input_dim=1)
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) 여기서부터 실행이 안 됨 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3.훈련
model.compile(loss='mse',optimizer='adam',metrics=['acc']) #metrics=['acc']는 진행되는 부분에 acc를 눈으로 보여주는 것
model.fit(x,y,epochs=30,batch_size=5)#batch_size가 낮다고 해서 꼭 좋은 loss값이 나오는 것은 아니다.  #계속 시행했을 때 acc=1.0나오면 좋은 값

#4.평가, 예측
loss,acc=model.evaluate(x,y,batch_size=5) #model.evaluate 기본적으로 loss, acc반환하는 함수

print("loss:",loss)
print("acc:",acc)






