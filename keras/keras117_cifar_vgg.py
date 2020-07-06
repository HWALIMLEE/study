# Normalizer
# 0과 1사이로 수렴시키겠다.
# 가중치 폭발 방지
# 계단함수-->sigmoid(상계문제)--->relu(음수값 자체가 반영이 안됨)(상게문제x)--->leaky relu(음수값 어느정도 인정, 음수값 무한으로 내려감)(상계문제)
# ---> elu(상계문제x)(음수 제한 걸어놓음)--->selu
# 계단함수에서 중간손실 막기 위해 sigmoid씀

### BatchNormalizer
# layer에 있는 거 normalization
# kernel = layer

"""
L1 규제: 가중치의 절대값 합
regularizer.l1(l=0.01)
L2 규제: 가중치의 제곱 합
regularizer.l2(l=0.01)

loss = L1*reduce_sum(abs(x)) 절댓갑 한 거 모두 더하겠다
loss = L2*reduce_sum(square(x)) 제곱한 규제값 모두 더하겠다

다음 레이어로 전달되는 loss값 축소하겠다는 의미

"""

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout, BatchNormalization, Activation
from keras.models import Model, Input, Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2, l1, l1_l2
from keras.applications import VGG16, VGG19
from keras.callbacks import EarlyStopping

# VGG16은 가중치 전달
# shape 잘 맞춰주기
vgg16 = VGG16()

vgg16 = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(32,32,3)
    # classifier_activation="softmax"
) 

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.reshape(50000,32,32,3).astype('float32')/255 #데이터 전처리
x_test=x_test.reshape(10000,32,32,3).astype('float32')/255

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)
vgg16.summary()
# sequential형으로 바꾸기
# kernel_regularizer 레이어에 명시
# MaxPooling은 엄밀하게 말하면 layer아님
model = Sequential()
model.add(vgg16)
model.add(Conv2D(32, kernel_size=3, padding="same")) # activation안 써줘도 linear적용 안되고, activation뒤에서 'relu'적용된다 
model.add(BatchNormalization()) 
# 원래 BatchNormalization목적은 activation이전에 작업해서 정리된 값 넘겨주는 것(Activation상위에 있을 때 좋다고 한다)
model.add(Activation('relu'))
model.add(Conv2D(32,kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))

"""
한 레이어
model.add(Conv2D(32,kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
"""


model.add(Conv2D(64, kernel_size=3, padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))


# model.add(Conv2D(128, kernel_size=3, padding="same"))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(128,kernel_size=3, padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding="same"))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

# 3. 훈련
# OneHotEncoding쓰지 않고 그냥 하고 싶을 때 loss='sparse_categorical_crossentropy'쓰면 된다. 
# 개인적인 취향
# 통상적으로 OneHotEncoding하고 있어야 함
early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=2)
model.compile(optimizer=Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['acc'])  #0.0004
hist = model.fit(x_train,y_train,
            epochs=20, batch_size=32, verbose=1,
            validation_split=0.3,callbacks=[early_stopping])

loss = model.evaluate(x_test, y_test)


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('acc 는 ', acc)
# print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

pred = np.argmax(model.predict(x_test[0:10]),axis=1)
print(pred)

# loss, acc 는  [0.5412761494636535, 0.8292999863624573]
