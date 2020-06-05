#Sequential()형으로 완성
#과제3

#하단에 주석으로 acc와 loss결과 명시하시오.

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


model=Sequential()
model.add(Dense(20,input_shape=(784,)))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
# model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 훈련
modelpath='./model/sample/fashion_mnist-{epoch:02d}-{val_loss:.4f}.hdf5' 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=1)
model.fit(x_train,y_train,epochs=10, batch_size=150,callbacks=[early_stopping,checkpoint],validation_split=0.2)

# model.save('./model/sample/fashion_mnist_model_save.h5')
# model.save_weights('./model/sample/fashion_mnist_save_weights.h5')


#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test,batch_size=50)

print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)
print("y_predict:",np.argmax(y_predict,axis=1))

