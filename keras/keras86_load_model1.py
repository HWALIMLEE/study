#85 번 복붙
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
# 0부터 255까지의 숫자
# 0은 하얀색, 255는 검은색

# plt.imshow(x_train[0],'gray') #흑백이면 gray #default값은 컬러...?
# plt.show()

# 28x28 사이즈만 출력
# 총 70,000개의 데이터

# y_train, y_test는 scalar 60000,10000개 벡터 1개
# 분류 모델로 가려면 10개로 분리(0-9)--->OneHotEncoding
# 0부터 9까지 숫자 70000개

# to_categorical쓰는 것이 편하다.
# to_categorical은 0이 없으면 0 넣어주고, 0있으면 자동으로 생김
# shape바꿀 필요 없음

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)
#원래 (60000,1)--->(60000,10)

# 데이터 전처리 2. 정규화
x_train=x_train.reshape(60000,28,28,1).astype('float32')/255 #--->MinMaxScaler #2차원에서 이미지 4차원으로 바꿔줌 CNN은 4차원(batch_size, 갸로, 세로, 채널)
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255  #--->정수형을 실수형으로 변환해준다(MinMax하기 위해서)

from keras.models import load_model
model=load_model('./model/model_test01.h5')
model.summary()

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
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# hist=model.fit(x_train,y_train,epochs=10,batch_size=100,validation_split=0.2)

#4. 평가, 예측
loss_acc=model.evaluate(x_test,y_test)
print("loss_acc:",loss_acc)


# import matplotlib.pyplot as plt 

# plt.figure(figsize=(10,6))

# #1.
# plt.subplot(2,1,1)  # (2행 1열의 1번째 그림을 그리겠다.) #두장 그릴 때는 무조건 subplot #(행, 열, 몇번째 넣을 건지(인덱스 1부터))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #x값은 자동으로 epoch가 될 것 #ex)plt.plot(x,y) # x 축 생략 #legend와 매치되는 것 순서대로
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# # plt.plot(hist.history['acc'],'b-')
# # plt.plot(hist.history['val_acc'],'p-')
# plt.grid() #모눈종이 처럼 보여주겠다
# # plt.legend(['loss','val_loss']) #범례(선에 대한 색깔과 설명)
# plt.legend(loc='upper right') #legend의 위치 (location) 명시 안하면 빈곳에 위치
# plt.title("loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")


# #2.
# plt.subplot(2,1,2)  # (2행 1열의 2번째 그림을 그리겠다.)
#  #x값은 자동으로 epoch가 될 것
# plt.plot(hist.history['acc'],'b-') #metrix값
# plt.plot(hist.history['val_acc'],'p-')
# # plt.plot(hist.history['val_loss'],'y-')
# # plt.plot(hist.history['val_loss'],'y-')

# plt.grid() #모눈종이 처럼 보여주겠다
# plt.legend(['acc','val_acc']) #범례(선에 대한 색깔과 설명)
# plt.title("acc")
# plt.ylabel("acc")
# plt.xlabel("epoch")


# plt.show() #넣지 않으면 출력이 안된다. 




y_predict=model.predict(x_test)
print("y_predict.shape:",y_predict.shape)
print("y_predict:",y_predict)

print(np.argmax(y_predict,axis=1))

"""
loss:0.05
acc:0.984 

keras85_save1.py 결과와 똑같다

fit 한 결과까지 저장되었다==가중치까지 저장되었다
"""

"""
훈련 전에 model save한 결과
loss: 0.05
acc: 0.988
"""