import numpy as np
import pandas as pd
from keras.utils import np_utils
datasets=np.load('./data/iris.npy')

print(datasets)
x=datasets[:150,0:3]
print(x)
y=datasets[:150,4]
print(y)

y=np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True, random_state=60,train_size=0.8)

from keras.models import load_model
model=load_model('./model/iris-31-0.1361.hdf5')

loss_acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss_acc:",loss_acc)

results=model.predict(x_test)
print("results:",np.argmax(results,axis=1))
