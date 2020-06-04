#95번을 불러와서 모델을 완성하시오.
import numpy as np
import pandas as pd
from keras.utils import np_utils
datasets=np.load('./data/iris.npy')

print(datasets)
"""
x=datasets[:150,0:3]
print(x)
y=datasets[:150,4]
print(y)

y=np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True, random_state=60,train_size=0.8)


from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 훈련
modelpath='./model/iris-{epoch:02d}-{val_loss:.4f}.hdf5'
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping=EarlyStopping(monitor='acc',patience=10,mode='aut')
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=1)
model.fit(x_train,y_train,epochs=50, batch_size=1,callbacks=[early_stopping,checkpoint],validation_split=0.2)

model.save('./data/iris_model_save.h5')

#모델 save할 때 데이터는 저장되지 않는다. 단지 input, output shape 만 저장될 뿐

loss_acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss_acc:",loss_acc)

result=model.predict(x_test)

print("result:",np.argmax(result,axis=1))

"""


