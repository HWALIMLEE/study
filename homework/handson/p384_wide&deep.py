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


housing = fetch_california_housing()

x_train_full,x_test,y_train_full,y_test = train_test_split(housing.data, housing.target)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_full,y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# model = Sequential()
# model.add(Dense(30,activation='relu',input_shape=x_train.shape[1:]))
# model.add(Dense(1))

# model.compile(loss='mse',optimizer='adam')
# hist = model.fit(x_train,y_train,epochs=20, validation_data=(x_valid,y_valid))
# mse_test = model.evaluate(x_test,y_test)
# x_new = x_test[:3]
# y_pred = model.predict(x_new)

# input_= Input(shape=x_train.shape[1:])
# hidden1 = Dense(30,activation='relu')(input)
# hidden2 = Dense(30,activation='relu')(hidden1)
# concat = Concatenate()([input_,hidden2])
# output = Dense(1)(concat)
# model = Model(inputs=[input_], outputs=[output])

# 깊은 경로
input_A = Input(shape=[5],name="wide_input")
input_B = Input(shape=[6],name="deep_input")
hidden1 = Dense(30,activation='relu')(input_B)
hidden2 = Dense(30,activation='relu')(hidden1)
concat = Concatenate()([input_A,hidden2])
# output = Dense(1,name="output")(concat)
# model = Model(inputs=[input_A,input_B],outputs=[output])

# model.compile(loss='mse',optimizer = SGD(lr=1e-3))

x_train_A, x_train_B = x_train[:,:5],x_train[:,2:]
x_valid_A, x_valid_B = x_valid[:,:5],x_valid[:,2:]
x_test_A, x_test_B = x_test[:,:5], x_test[:,2:]
x_new_A, x_new_B = x_test_A[:3],x_test_B[:3]

# history = model.fit((x_train_A, x_train_B),y_train, epochs=20,
#                     validation_data=((x_valid_A, x_valid_B),y_valid))
# mse_test = model.evaluate((x_test_A,x_test_B),y_test)

# y_pred = model.predict((x_new_A, x_new_B))

output = Dense(1,name="main_output")(concat)
aux_output = Dense(1,name="aux_output")(hidden2)
model = Model(inputs=[input_A, input_B], outputs=[output,aux_output])

model.compile(loss=["mse","mse"],loss_weights=[0.9,0.1],optimizer='sgd')

# 주 출력과 보조 출력이 같은 것을 예측해야 하므로 동일한 레이블 사용
history = model.fit([x_train_A, x_train_B],[y_train,y_train],epochs=20,
                    validation_data=([x_valid_A, x_valid_B],[y_valid,y_valid]))
            
total_loss,main_loss,aux_loss = model.evaluate([x_test_A, x_test_B],[y_test,y_test])
y_pred_main,y_pred_aux = model.predict([x_new_A, x_new_B])

print(y_pred_main)
print(y_pred_aux)