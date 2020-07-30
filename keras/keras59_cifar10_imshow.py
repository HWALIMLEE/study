from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

#데이터 구조 알아보기
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print("x_train[0]:",x_train[0])
print("y_train[0]:",y_train[0])

print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

plt.imshow(x_train[0])
plt.show()

#(50000,1)--->(50000,10)
#(32,32,3)으로 바꿔줌 3은 컬러
