import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU,BatchNormalization, Activation
from keras.datasets import fashion_mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l1, l2

(x_train_full,y_train_full), (x_test,y_test) = fashion_mnist.load_data()

x_valid,x_train = x_train_full[:5000]/255.0, x_train_full[5000:] / 255.0
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]
x_test = x_test/255.0


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(BatchNormalization())
model.add(Dense(300,kernel_initializer='he_normal',use_bias=False, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(100, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('elu'))  # model.add(Activation='elu') (x)
model.add(Dense(10,activation='softmax'))

model.summary()

# 배치 정규화 층은 입력마다 이동 파라미터를 포함하기 때문에 이전 층에서 편향을 뺄 수 있습니다. 


# checkpoint = ModelCheckpoint("./model/my_keras_model.h5")
def exponential_decay_fn(epoch):
    return 0.01*0.1**(epoch/20)
lr_scheduler = LearningRateScheduler(exponential_decay_fn)

# 거듭제곱 기반 스케줄링
optimizer = SGD(lr=0.001,decay=1e-4)
model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=30,
                validation_data=(x_valid,y_valid),callbacks=[lr_scheduler])


import pandas as pd
import matplotlib.pyplot as plt




# pd.DataFrame(hist.history).plot(figsize=(8,5))
# plt.grid()
# plt.gca().set_ylim(0,1)
# plt.show()

loss,acc = model.evaluate(x_test,y_test)
print("loss:",loss)
print("acc:",acc)

###l2
# loss: 0.5786471096992493
# acc: 0.8859000205993652