import numpy as np
import pandas as pd
from keras.models import Sequential
from pandas import DataFrame
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

train=pd.read_csv("C:\\Users\\hwalim\\kaggle\\train.csv")
test=pd.read_csv("C:\\Users\\hwalim\\kaggle\\test.csv")

print("train.shape:",train.shape)
print("test.shape:",test.shape)

#우리가 추출하고 알아내고 싶은 숫자
y_train=train["label"]
print(y_train)

#label제거
x_train=train.drop(labels=["label"],axis=1)

#y변환
y_train=np_utils.to_categorical(y_train)
print("y_train:",y_train)

x_train=x_train.values.reshape(-1,28,28,1).astype('float32')/255 #MinMaxScaler
test=test.values.reshape(-1,28,28,1).astype('float32')/255

# print(train.shape)
# print(test.shape)

model=Sequential()
model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(30,(2,2),padding="same",activation='relu'))
model.add(Conv2D(50,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
model.add(Conv2D(20,(2,2),padding="same",activation='relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(30,(2,2),padding="same",activation='relu'))
model.add(Conv2D(50,(2,2),padding="same",activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(10,activation="softmax"))

#3.훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='acc',patience=10,mode='aut')
model.fit(x_train,y_train,epochs=100,batch_size=100,callbacks=[early_stopping])

results=model.predict(test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name="label")

submission=pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)
submission.to_csv("D:\\STUDY\\kaggle\\cnn_mnist_dataset.csv",index=False)