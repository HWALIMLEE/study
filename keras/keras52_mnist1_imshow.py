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
#0부터 255까지의 숫자
#0은 하얀색, 255는 검은색

plt.imshow(x_train[0],'gray') #흑백이면 gray 
plt.show()

print(x_train[0].shape) #28x28 사이즈만 출력
#총 70,000개의 데이터
