import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist #datasets은 예제파일 들어가 있음
(x_train,y_train),(x_test,y_test)=mnist.load_data() #load_data에 train, test분리 되어 있음

print("x_train:",x_train[0]) #x의 첫번째
print("y_train:",y_train[0]) #y의 첫번째
print("x_train.shape:",x_train.shape) #batch_size, height, width
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape) #60000개의 스칼라  #y의 dimesion은 1
print("y_test.shape:",y_test.shape)
print(x_train[0].shape)

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)
#원래 (60000,1)--->(60000,10)

# 데이터 전처리 2. 정규화
# 현재 데이터 255까지--->0과 1로 바꿔줌(MinMax)
# 실수형으로 변환 
x_train=x_train.reshape(60000,28,28,1).astype('float32')/255 #--->MinMaxScaler #2차원에서 이미지 4차원으로 바꿔줌 CNN은 4차원(batch_size, 갸로, 세로, 채널)
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255  #--->정수형을 실수형으로 변환해준다(MinMax하기 위해서)


"""
정말 간단하게 말하면 float32는 32비트를 사용하고 float64는 64비트를 사용한다는 것인데, 
이는 float64의 메모리 사용량이 두 배라는 것을 의미하며 따라서 연산속도가 느려질 수 있는 것을 의미한다.
하지만, float64는 float32에 비해 훨씬 정확하게 숫자를 나타낼 수 있으며 훨씬 큰 숫자를 저장할 수 있다는 장점을 가진다.
"""

# CNN을 함수형으로 만들기
# 모델구성
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout

input1=Input(shape=(28,28,1))
dense1=Conv2D(10,(2,2),activation='relu')(input1)
dense2=Conv2D(20,(2,2),activation='relu')(dense1)
dense3=Conv2D(50,(2,2),activation='relu')(dense2)
dense4=Conv2D(30,(3,3),activation='relu')(dense3)
dense5=Dropout(0.2)(dense4)

output1=Conv2D(50,(2,2),activation='relu')(dense5)
output2=Conv2D(30,(2,2),activation='relu')(output1)
output3=MaxPooling2D(pool_size=2)(output2)
output4=Flatten()(output3)
output5=Dense(10,activation="softmax")(output4)

model=Model(input=input1,output=output5)

model.summary()
#<Dnn>
# model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))
# model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.2)) #이 레이어에 있는 노드의 20%제거(아무데나 써주고 싶은 데 써도 된다.)
# model.add(Conv2D(30,(2,2),padding="same",activation='relu'))
# model.add(Conv2D(30,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(30,(2,2),padding="same",activation='relu'))
# model.add(Conv2D(50,(2,2),padding="same",activation='relu'))

# model.add(MaxPooling2D(pool_size=2))


# model.add(Flatten())
# model.add(Dense(10,activation='softmax')) # 마지막 dense모델에 다중분류는 activation='softmax'가 들어가야 한다. 위에 conv2D에 들어가는 것 아님

model.summary()


# 3. 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=100)

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test,batch_size=100)  #loss, metrics에 집어넣은 값
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict.shape:",y_predict.shape)
print("y_predict:",y_predict)

print(np.argmax(y_predict,axis=1))
