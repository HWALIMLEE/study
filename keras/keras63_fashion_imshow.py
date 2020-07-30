import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
#과제1
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

print("x_train[0]:",x_train[0])

plt.imshow(x_train[0])
plt.show()
