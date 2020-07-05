from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.datasets import fashion_mnist
from keras.optimizers import Adam, SGD

(x_train_full,y_train_full), (x_test,y_test) = fashion_mnist.load_data()

x_valid_B,x_train_B = x_train_full[:5000]/255.0, x_train_full[5000:] / 255.0
y_valid_B,y_train_B = y_train_full[:5000],y_train_full[5000:]
x_test_B = x_test/255.0
y_test_B = y_test


from keras.models import load_model
import keras
model_A = load_model("./model/my_keras_model.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(Dense(1,activation='sigmoid',name='new_dense'))

model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

print(x_train_B.shape)
print(y_train_B.shape)

for layer in model_B_on_A.layers[:-1]:
    layer.trainable=False
model_B_on_A.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['acc'])

history = model_B_on_A.fit(x_train_B, y_train_B, epochs=4,validation_data = (x_valid_B,y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable=True

optimizer = SGD(lr=1e-4)
model_B_on_A.compile(loss='binary_crossentropy',optimizer = optimizer, metrics=['acc'])

history = model_B_on_A.fit(x_train_B, y_train_B, epochs=16, validation_data=(x_valid_B, y_valid_B))

loss, acc = model_B_on_A.evaluate(x_test_B, y_test_B)

print("loss:",loss)
print("acc:",acc)
