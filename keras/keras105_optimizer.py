#1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))


from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Nadam
                                               # loss    pred
# optimizer = Adam(learning_rate=0.001)        # 0.05, 3.407
# optimizer = Adagrad(learning_rate=0.001)     # 8.292, -0.203
optimizer = SGD(learning_rate=0.001)         # 0.08,  3.351
# optimizer = Adadelta(learning_rate=0.001)    # 7.775, -0.063
# optimizer = Nadam(learning_rate=0.001)       # 0.115, 3.15887
# optimizer = RMSprop(learning_rate=0.001)       # 4.46,  3.502

model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])

model.fit(x,y,epochs=100)

loss = model.evaluate(x,y)
print("loss:",loss)

pred1 = model.predict([3.5])
print(pred1)
