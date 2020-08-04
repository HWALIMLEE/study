from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Flatten
import matplotlib.pyplot as plt
import random

# cnn으로 오토인코더 구성하시오.
# 딥하게 구성
# hidden_layer추가해보자
# 오토 인코더는 비선형
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3),padding='same',input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=30, kernel_size=(3,3),padding='same',activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same',activation='sigmoid'))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(units=1, activation='sigmoid'))
    model.summary()
    return model


from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train,y_train = train_set
x_test,y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape)
print(x_test.shape)


model = autoencoder(hidden_layer_size=154) 

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) 

model.fit(x_train,x_train, epochs=20)

output = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7, ax8, ax9, ax10)) = plt.subplots(2,5,figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다. 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


#오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT",size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()