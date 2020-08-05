# noise를 주어 없애는 과정으로부터 학습
# CNN으로 바꾸기

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
import matplotlib.pyplot as plt
import random
import numpy as np

# 오토 인코더는 비선형
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3,3),padding='valid',input_shape=(28,28,1),activation='relu')) # decoder에서 크기가 커져야 한다. 
    model.add(Conv2D(filters=30, kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(filters=10, kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2DTranspose(filters=30,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2DTranspose(filters=10,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=(3,3), padding='valid',activation='sigmoid')) # Conv2DTranspose하면 줄인만큼 다시 늘려주는 작업
    model.summary()
    return model


from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train,y_train = train_set
x_test,y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)
x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape)
print(x_test.shape)

# 노이즈 주기
x_train_noised = x_train + np.random.normal(0,0.1, size = x_train.shape) # 0: 평균, 0.5: 표준편차
x_test_noised = x_test + np.random.normal(0,0.1,size = x_test.shape)

# 0부터 1사이를 넘어갈 수 있다. 조정 필요
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 배열에서 0보다 작은 수를 0으로 변환
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)



model = autoencoder(hidden_layer_size=32) # n_components숫자 => hidden_layer_size숫자

# model.compile(optimizer='adam',loss='mse',metrics=['acc']) #loss: 0.01, acc: 0.01(hidden_layer_size=32개일 때)               # sigmoid때문에 0과 1 사이로 수렴
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) # loss: 0.09, acc: 0.8                             # binary_crossentropy때문에 0과 1사이로 수렴

# loss는 위에 것이 더 작고, acc는 밑에 것이 더 높다
# 그러나 우리는 loss를 기준으로 봐야 한다. 

model.fit(x_train_noised,x_train, epochs=10) #잡음이 없는 것과 잡음이 있는 것을 비교

output = model.predict(x_test_noised)

# print("output.shape:",output.shape[0])
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7, ax8, ax9, ax10),
                    (ax11, ax12, ax13, ax14, ax15)) = \
                                    plt.subplots(3,5,figsize=(20,7))


# 원본, 노이즈, 아웃풋
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다. 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])



# 아웃풋
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# 잡음 부분이 뭉그러졌다.
# 사람 이미지를 넣으면 흐릿하게 나온다. >>> GAN은 이 과정을 반복해가면서 뚜렷한 이미지로 바꿔준다.
# 오토인코더와 GAN은 다른 방식 꼭 같이 쓰는 거 아니다. 
