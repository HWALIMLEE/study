import numpy as np
from sklearn.datasets import load_iris
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#1. 데이터
"""
data : x값
target : y값
"""

dataset=load_iris()
print("dataset:",dataset)

x=dataset.data
y=dataset.target
print("x:",x)
print("y:",y)
print("x.shape:",x.shape)
print("y.shape:",y.shape)

x=StandardScaler().fit_transform(x)

pca=PCA(n_components=2) #주성분 개수
trans_x=pca.fit_transform(x)

print("trans_x:",trans_x)
print("trans_x.shape:",trans_x.shape)


x_train,x_test,y_train,y_test=train_test_split(trans_x,y,train_size=0.8)
print(x_train.shape)
print(x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape",y_test.shape)

x_train=x_train.reshape(120,2)
x_test=x_test.reshape(30,2)
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)

print("y_train:",y_train)

model=Sequential()
model.add(Dense(10,input_shape=(2,),activation='relu'))
model.add(Dense(20))
model.add(Dense(20,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()
modelpath='./model/sample/iris-{epoch:02d}-{val_loss:.4f}.hdf5' 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',
                            save_best_only=True,save_weights_only=False,verbose=1) 
model.fit(x_train,y_train,epochs=10,batch_size=1,callbacks=[early_stopping,checkpoint],validation_split=0.2)

# model.save('./model/sample/iris_model_save.h5')
# model.save_weights('./model/sample/iris_save_weights.h5')

loss,acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("acc:",acc)

y_predict=model.predict(x_test)

print("y_predict:",np.argmax(y_predict,axis=1))


from sklearn.metrics import mean_squared_error as mse
def RMSE(y_test,y_predict):
    np.sqrt(mse(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)




