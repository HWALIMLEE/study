#1. 데이터
import numpy as np
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([11,12,13,14,15])
y_test=np.array([11,12,13,14,15]) #테스트 데이터는 모델 성능에 영향 미치지 않는다. 
x_pred=np.array([16,17,18])


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
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train,y_train,epochs=50,batch_size=1)#batch_size가 낮다고 해서 꼭 좋은 loss값이 나오는 것은 아니다. #계속 시행했을 때 acc=1.0나오면 좋은 값 #훈련train

#4.평가
loss,mse=model.evaluate(x_test,y_test,batch_size=1) #model.evaluate 기본적으로 compile에서 설정한 loss, metrics반환하는 함수 #evaluate는 test값 평가 #그런데 같은 데이터 값으로 다시 평가했음 #과적합
print("loss:",loss)
print("mse:",mse)

#5.예측
y_pred=model.predict(x_pred)
print("y_predict:",y_pred)
#훈련데이터와 평가용 데이터는 같은 데이터 쓰게 되면 안된다. 과적합