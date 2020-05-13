#1. 데이터
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10]) 
x_pred=np.array([11,12,13])
#y_pred를 예측할것

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
model.compile(loss='mse',optimizer='adam',metrics=['mse']) #metrics=['acc']는 진행되는 부분에 acc를 눈으로 보여주는 것 #회귀모델을 보고 싶은데 metrics=['acc']를 설정해놔서 잘못 된 것/ metrics=['mse']로 바꿔야함
#metrics=['acc']로 해놓으면 회귀모델인데도 불구하고 0또는 1로 나옴..
model.fit(x,y,epochs=50,batch_size=1)#batch_size가 낮다고 해서 꼭 좋은 loss값이 나오는 것은 아니다. #계속 시행했을 때 acc=1.0나오면 좋은 값 #훈련train

#4.평가, 예측
loss,mse=model.evaluate(x,y,batch_size=5) #model.evaluate 기본적으로 compile에서 설정한 loss, metrics반환하는 함수 #evaluate는 test값 평가 #그런데 같은 데이터 값으로 다시 평가했음 #과적합
print("loss:",loss)
print("mse:",mse)

y_pred=model.predict(x_pred)
print("y_predict:",y_pred)
#훈련데이터와 평가용 데이터는 같은 데이터 쓰게 되면 안된다. 과적합
