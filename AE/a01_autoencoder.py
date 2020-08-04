import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape) # (60,000,28,28)
print(y_train.shape) # (60,000,)
print(x_test.shape)  # (10,000,28,28)
print(y_test.shape)  # (10,000,)

# autoencoder니까 one.hot.encoding할 필요 없음
"""
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)
"""

x_train=x_train.reshape(60000,784).astype('float32')/255 
x_test=x_test.reshape(10000,784).astype('float32')/255
print(x_train.shape)

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
# 중간 노드의 개수 파악
# 특성을 추출하고 차원을 줄이는 것(pca)
decoded = Dense(784, activation='sigmoid')(encoded) # 0부터 1사이니까(정규화시켜서), sigmoid써준다. # 784개 이미지를 다시 784개 이미지로 복원하는 거기 때문에, softmax아님, 다중분류 아님

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2) # 앞뒤가 똑같은 오토인코더~~

decoded_imgs = autoencoder.predict(x_test) # predict


import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 필요 없는 값들 제거하고 특성 추출
# 필요 없는 부분들 == 배경(이미지에서)
# autoencoder쓸 것인지, GAN을 쓸 것인지 고민
