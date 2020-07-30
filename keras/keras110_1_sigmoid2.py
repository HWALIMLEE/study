# 동일한 데이터 , 한개의 모델
# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10,input_shape=(1,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(100,activation='sigmoid')) # activation='linear'가 기본값으로 들어가게 된다. / 'tanh'도 있을 수 있음/ 'sigmoid'도 쓸 수 있다.
# 하지만 중간에 sigmoid쓰지 않는 게 좋다.
# 'relu'를 많이 쓴다(0이하는 모두 0으로, 0이 넘어가면 선형으로 해준다.)
# 중간에 sigmoid한번 쓰게 되면 튀는 값들이 한번 수렴될 수 있다.
model.add(Dense(1,activation='sigmoid')) # sigmoid는 무조건 0과 1사이

model.summary()

#3. 컴파일, 훈련
model.compile(loss = ['binary_crossentropy'],optimizer = 'adam',
                        metrics=['mse','acc'])
# binary_crossentropy===>
# 분기 시킨다
# loss값 7개 나온다.
model.fit(x_train,y_train,epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_train,y_train)
print("loss:",loss)
x1_pred = np.array([11,12,13,14])
# 11,12,13,14의 회귀값과 분류값 동시에 나옴
# 이 값이 어떻게 나오는 지 분석

y_pred = model.predict(x1_pred)
print("y_pred:",y_pred)

# 전체 dense의 loss = dense5의 loss + dense8의 loss
# 지표 1개로 모델 1개 돌리는 게 제일 Best
