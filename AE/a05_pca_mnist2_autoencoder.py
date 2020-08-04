# pca로 만든 컬럼 모델에 집어넣을 수 있다. 
# input_dim = 154로 모델을 만드시오

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape) # (60,000,28,28)
print(y_train.shape) # (60,000,)
print(x_test.shape)  # (10,000,28,28)
print(y_test.shape)  # (10,000,)

# 데이터 전처리
x_train=x_train.reshape(60000,784).astype('float32')/255
x_test=x_test.reshape(10000,784).astype('float32')/255

X = np.append(x_train,x_test,axis=0)

pca = PCA(n_components=154)
X = pca.fit_transform(X)

print(X.shape) #(70,000, 154)

# autoencoder니까 one.hot.encoding할 필요 없음
x2_train = X[:60000]
x2_test = X[60000:]

print(x2_train.shape)
print(x2_test.shape)


input_img = Input(shape=(154,))
encoded = Dense(32, activation='relu')(input_img)
# 중간 노드의 개수 파악
# 특성을 추출하고 차원을 줄이는 것(pca)
decoded = Dense(154, activation='sigmoid')(encoded) # 0부터 1사이니까(정규화시켜서), sigmoid써준다. # 다중분류에서도 sigmoid써줄 수 있음

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
# autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.fit(x2_train, x2_train, epochs=50, batch_size=50, validation_split=0.2) # 앞뒤가 똑같은 오토인코더~~

decoded_imgs = autoencoder.predict(x2_test) # predict

loss, acc = autoencoder.evaluate(x2_test,decoded_imgs, batch_size=50)

print("loss:",loss)
print("acc:",acc)

"""
loss: 3.1720477634906217e-05
acc: 0.9996083
"""
