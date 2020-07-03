import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

(x_train_full,y_train_full), (x_test,y_test) = fashion_mnist.load_data()

x_valid,x_train = x_train_full[:5000]/255.0, x_train_full[5000:] / 255.0
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]
x_test = x_test/255.0


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=0.01),metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=30,
                validation_data=(x_valid,y_valid))


import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(hist.history).plot(figsize=(8,5))
plt.grid()
plt.gca().set_ylim(0,1)
plt.show()