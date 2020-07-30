# batch_size, time_steps, feature
# (784,1) 로 바꾸기 =(28,28)-->차원을 낮추어야 하니까/ 어떤 것이 더 괜찮게 나오는지 확인해보기
# 784개를 한개씩 자르는 거(784,1), 28개를 한번에 28개씩 자르는거(28,28)
# shape만 조작하지 않으면 된다. 

# mnist-cnn(keras54)
# mnist_dnn(keras56)
# mnist_lstm(keras58)


#데이터 전처리 1. 원핫인코딩
from keras.datasets import mnist #datasets은 예제파일 들어가 있음
(x_train,y_train),(x_test,y_test)=mnist.load_data() #load_data에 train, test분리 되어 있음

import numpy as np
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)
#원래 (60000,1)--->(60000,10)


# 데이터 전처리 2. 정규화
# 현재 데이터 255까지--->0과 1로 바꿔줌(MinMax)
# 실수형으로 변환 
x_train=x_train.reshape(60000,49,16).astype('float32')/255 #--->MinMaxScaler #2차원에서 이미지 4차원으로 바꿔줌 CNN은 4차원(batch_size, 갸로, 세로, 채널)
x_test=x_test.reshape(10000,49,16).astype('float32')/255  #--->정수형을 실수형으로 변환해준다(MinMax하기 위해서)
# Max값 모르면 바로 MinMax쓰고
# 알면 위에처럼 그냥 나누면 됨

"""
정말 간단하게 말하면 float32는 32비트를 사용하고 float64는 64비트를 사용한다는 것인데, 
이는 float64의 메모리 사용량이 두 배라는 것을 의미하며 따라서 연산속도가 느려질 수 있는 것을 의미한다.
하지만, float64는 float32에 비해 훨씬 정확하게 숫자를 나타낼 수 있으며 훨씬 큰 숫자를 저장할 수 있다는 장점을 가진다.
"""

#실습...!
# 1. 255.
# 2. 255.0 으로 해보기

#모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout,LSTM

# 활성화함수 넣어주기
# 과적합을 없애는 방법--dropout
# 성능이 좋아지는 경우도 있고, 안 좋아질 수도 있다
# LSTM은 maxpooling쓸 수 없다.
# flatten()도 또한 쓸 수 없다

model=Sequential()
model.add(LSTM(10,input_shape=(49,16)))
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
# model.add(Flatten()) #flatten()을 시켜주어야 함
model.add(Dense(10,activation='softmax'))


model.summary()


# 3. 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
#binary_crossentropy 써보기: 

# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# 분류 모델에서는 accuracy
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실함수는 평균제곱오차 크로스 엔트로피 함수 사용--->다중분류
# 이중분류일떄는 binary_crossentropy

model.fit(x_train, y_train, epochs=20, batch_size=100)

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test,batch_size=100)  #loss, metrics에 집어넣은 값
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict.shape:",y_predict.shape)
print("y_predict:",y_predict)

print(np.argmax(y_predict,axis=1))

#최종값: 