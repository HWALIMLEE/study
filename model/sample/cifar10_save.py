import numpy as np
from keras.models import Model, Input
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape)
x_train=x_train.reshape(50000,3072).astype('float32')/255
x_test=x_test.reshape(10000,3072).astype('float32')/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

input1=Input(shape=(3072,))
dense1=Dense(30,activation='relu')(input1)
dense2=Dense(10,activation='relu')(dense1)
dense3=Dense(20,activation='relu')(dense2)
dense4=Dense(15,activation='relu')(dense3)

output1=Dense(20,activation='relu')(dense4)
output2=Dense(10,activation='relu')(output1)
output3=Dense(10,activation='softmax')(output2)

model=Model(input=input1, output=output3)

model.summary()

modelpath='./model/sample/cifar10-{epoch:02d}-{val_loss:.4f}.hdf5' 

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='atu')
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',
                            save_best_only=True,save_weights_only=False,verbose=1) 
hist=model.fit(x_train,y_train,epochs=20,batch_size=100,validation_split=0.2,callbacks=[early_stopping,checkpoint])


# model.save('./model/sample/cifar10_model_save.h5')
# model.save_weights('./model/sample/cifar10_save_weight.h5')

loss,acc=model.evaluate(x_test,y_test,batch_size=100)

y_predict=model.predict(x_test)

print(np.argmax(y_predict,axis=1))

