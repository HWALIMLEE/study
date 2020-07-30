from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras. layers import Dense, Embedding, LSTM, Flatten

#1. 데이터
(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=10000, test_split = 0.2)
# 가장 빈도수가 많은 것부터 1000번쨰까지

print(x_train.shape, x_test.shape)  #(8982,), (2246,)
print(y_train.shape, y_test.shape)  #(8982,), (2246,)

print(x_train[0])
print(y_train[0])

#list는 shape구할 수 없다

print(len(x_train[0])) # 87

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리:",category) #46개

# y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

# 주간과제: groupby()의 사용법 숙지할 것

########################################

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

####다시 해보기
x_train = pad_sequences(x_train, maxlen=100, padding='pre') #maxlen - 최대값을 100으로 잡는다. / truncating - ex)100개가 넘는 문자열 자를 때(default = 앞에서부터)
# 여기서는 87개이기 때문에 truncating 필요 없다
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

print(len(x_train[0]))  #87
print(len(x_train[-1])) #105


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(8982,100), (2246,100)

#2. 모델
model = Sequential()
# model.add(Embedding(1000, 100, input_length=100))
model.add(Embedding(1000,100))
# parameter 개수가 너무 커져도 좋지 않고 알아서 판단해야함
# word+_size 통상적으로 들어가는 단어 개수 넣어주는 것
# output*word_size=첫번째 파라미터 개수
# input_length안 써주면 x_train의  input_length 값을 자연스럽게 가져옴
# Embedding은 3차
model.add(LSTM(100))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc']) #sparse_categorical_crossentropy

history = model.fit(x_train,y_train,batch_size=100,epochs=10, validation_split=0.2)

acc = model.evaluate(x_test,y_test)[1]
print("acc:",acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss,marker=',',c='red',label='TestSet loss')
plt.plot(y_loss,marker=',',c='blue',label='Trainset loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 8000개로 훈련, 2000여개로 맞추는 것