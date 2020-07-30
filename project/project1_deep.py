import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model,Input,Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

data = pd.read_csv("./project/data/data_company.csv",index_col=0, header=0, thousands=',')
data = data.fillna(0)
test_data = pd.read_csv("./project/test_data/data_company_test.csv",index_col=0, thousands=',',header=None)
test_data = test_data.fillna(0)
print(data)
print(test_data)

np.save("./project/data_company.npy",arr = data)
np.save("./project/data_company_test.npy",arr = test_data)
data = np.load("./project/data_company.npy", allow_pickle=True)
test_data = np.load("./project/data_company_test.npy",allow_pickle=True)

x = data[0:,0:-1].astype('int64')
y = data[0:,-1]
encoder=LabelEncoder()
encoder.fit(y)
y_encode = encoder.transform(y)
test_data = test_data.astype('int64')
# y_encode=np_utils.to_categorical(y_encode)

print("x:",x)
print("y:",y_encode)
print("test_data:",test_data)

print("x.shape:",x.shape) #(80,135)
print("y.shape:",y.shape) #(80,)


#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y_encode,test_size=0.2,random_state=60)

print(x_train.shape)
print(x_test.shape)

scaler1=StandardScaler()
scaler1.fit(x_train)
x_train_scaled = scaler1.fit_transform(x_train)

print("x_train_scaled:",x_train_scaled)
print("y_train:",y_train)

#2. 모델
model = Sequential()
model.add(Dense(512,input_dim=135))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

#3. 훈련, 평가
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

# early_stopping=EarlyStopping(monitor='acc',mode='aut',patience=10)
model.fit(x_train_scaled,y_train, epochs=100,batch_size=1,validation_split=0.1)

y_predict=model.predict(x_test)
loss,acc=model.evaluate(x_test,y_test)

print("loss:",loss)
print("acc:",acc)
