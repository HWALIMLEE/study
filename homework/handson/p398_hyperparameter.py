import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from keras.datasets import fashion_mnist
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

x_train_full,x_test,y_train_full,y_test = train_test_split(housing.data, housing.target)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_full,y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

x_new = x_test[:3]


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))
    optimizer = SGD(lr=learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    return model


keras_reg = KerasRegressor(build_model)

keras_reg.fit(x_train,y_train,epochs=100,
                validation_data=(x_valid,y_valid),
                callbacks=[EarlyStopping(patience=10)])


mse_test = keras_reg.score(x_test,y_test)
y_pred = keras_reg.predict(x_new)

print(y_pred)