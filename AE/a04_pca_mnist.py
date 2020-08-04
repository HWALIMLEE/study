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

# 데이터 전처리
x_train=x_train.reshape(60000,784).astype('float32')/255 
x_test=x_test.reshape(10000,784).astype('float32')/255

X = np.append(x_train, x_test, axis=0)

print(X.shape) # (70000, 784)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

best_n_components = np.argmax(cumsum>=0.95) + 1 # 인덱스는 0부터 시작하니까 1더해주기 # 154개는 날려도 된다
print(best_n_components)
