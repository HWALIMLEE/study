import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

train_features=pd.read_csv("./data/dacon/comp3/train_features.csv",header=0,index_col=0)
train_target=pd.read_csv("./data/dacon/comp3/train_target.csv",header=0,index_col=0)
test_features=pd.read_csv("./data/dacon/comp3/test_features.csv",header=0,index_col=0)

#바로 LSTM구성 가능 shpae
print("train_features.shape:",train_features.shape) #(1050000,5)
print("train_target.shape:",train_target.shape)   #(2800,4)
print("test_features.shape:",test_features.shape) #(262500,5)

#결측값은 없다
print(train_features.isnull().sum()) #없음
print(train_target.isnull().sum()) #없음
print(test_features.isnull().sum()) #없음

#저장하기
np.save('./data/comp3_train_features.npy',arr=train_features)
np.save('./data/comp3_train_target.npy',arr=train_target)
np.save('./data/comp3_test_features.npy',arr=test_features)

#불러오기
train_data=np.load('./data/comp3_train_features.npy')
train_target=np.load('./data/comp3_train_target.npy')
test_data=np.load('./data/comp3_test_features.npy')
train_data=train_data[0:,1:]
test_data=test_data[0:,1:]
print(train_data)

train_data=StandardScaler().fit_transform(train_data)
test_data=StandardScaler().fit_transform(test_data)
print(train_data.shape)

train_data=train_data.reshape(2800,375,4)
test_data=test_data.reshape(700,375,4)

x_train,x_test,y_train,y_test=train_test_split(train_data,train_target,test_size=0.2)


"""
#데이터 자르기
def split_xy(dataset,time_steps,y_column):
    x,y=list(),list()
    for i in range(len(dataset)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_column-1
        if y_end_number>len(dataset):
            break
        tmp_x=dataset[i:x_end_number,0:]
        tmp_y=dataset[x_end_number-1:y_end_number,-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x),np.array(y)
x,y=split_xy(train_data,4,4)
print("x:",x)
print("y:",y)
print("x.shape:",x.shape)
print("y.shape:",y.shape)
"""

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=Sequential()
model.add(Conv1D(10,2,input_shape=(375,4),activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(4,activation='relu'))


model.compile(loss='mse',optimizer='adam',metrics=['mse'])

model.fit(x_train,y_train,epochs=100,batch_size=100)

y_predict=model.predict(x_test)
result=model.predict(test_data)

print(y_predict.shape)
print(y_predict)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_predict)
print("mse:",mse)


a = np.arange(2800,3500)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp3/sample_submission1_1.csv", header = ["X","Y","M","V"], index = True, index_label="id" )