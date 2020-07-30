from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras. layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam

#1. 데이터
# 이 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1을 부정인 경우 0으로 표시한 레이블로 구성된 데이터입니다.
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000000) # imdb 는 test_split안됨 왜?
# 가장 빈도수가 많은 것부터 1000번쨰까지
# y가 0과 1로 이루어져 있음

print(x_train.shape, x_test.shape)  #(25000,) (25000,)
print(y_train.shape, y_test.shape)  #(25000,) (25000,)

print(x_train[0])
print(y_train[0])

#list는 shape구할 수 없다

print(len(x_train[0])) #218

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리:",category) #2

# y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
print(y_train_pd)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)




# 주간과제: groupby()의 사용법 숙지할 것

########################################

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

####다시 해보기
x_train = pad_sequences(x_train, maxlen=111, padding='pre') #maxlen - 최대값을 100으로 잡는다. / truncating - ex)100개가 넘는 문자열 자를 때(default = 앞에서부터)
# 여기서는 87개이기 때문에 truncating 필요 없다
x_test = pad_sequences(x_test, maxlen=111, padding='pre') #maxlen은 y 의 열의 개수

print(len(x_train[0]))  
print(len(x_train[-1])) 


print(x_train.shape, x_test.shape)

#2. 모델
model = Sequential()
# model.add(Embedding(2000, 100, input_length=111))
model.add(Embedding(2000,100))
# parameter 개수가 너무 커져도 좋지 않고 알아서 판단해야함
# word+_size 통상적으로 들어가는 단어 개수 넣어주는 것
# output*word_size=첫번째 파라미터 개수
# input_length안 써주면 x_train의  input_length 값을 자연스럽게 가져옴
# Embedding은 3차
model.add(Conv1D(64,5,padding='valid',strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Conv1D(64,2, padding='same',strides=1,kernel_initializer='he_normal',kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=4))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Conv1D(64,2, padding='same',strides=1,kernel_initializer='he_normal'))
model.add(MaxPooling1D(pool_size=4))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Conv1D(64,2, padding='same',strides=1,kernel_initializer='he_normal'))
model.add(MaxPooling1D(pool_size=4))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(Conv1D(64,2, padding='same',strides=1,kernel_initializer='he_normal'))
model.add(MaxPooling1D(pool_size=4))
model.add(BatchNormalization())
model.add(Activation('selu'))

model.add(LSTM(4))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))
# embedding은 LSTM 없으면 안돌아감...쩝
model.summary()

optimizers = SGD(lr=0.001, momentum=0.9, nesterov=True)
early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=100)
model.compile(loss='binary_crossentropy',optimizer=optimizers,metrics=['acc']) #sparse_categorical_crossentropy

history = model.fit(x_train,y_train,batch_size=100,epochs=10000, validation_split=0.2,callbacks=[early_stopping])

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

#1. imdb검색해서 데이터 내용 확인.
#2. word_size전체데이터 부분 변경해서 최상값 확인
#3. groupby ()사용법 숙지
#4. 숫자를 문자로 / pd.get_dummies
