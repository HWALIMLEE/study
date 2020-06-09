import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

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

# train_data=StandardScaler().fit_transform(train_data)
print(train_data.shape)

train_data=train_data.reshape(2800,375,4)
test_data=test_data.reshape(700,375,4)

x_train,x_test,y_train,y_test=train_test_split(train_data,train_target,test_size=0.2)

print(x_train.shape) #(2240,375,4)
print(x_test.shape)  #(560,375,4)
print(y_train.shape) #(2240,4)
print(y_test.shape)  #(560,4)

x_train=x_train.reshape(2240,375*4)
x_test=x_test.reshape(560,375*4)
test_data=test_data.reshape(700,375*4)

#pipe매개변수 'model'써주기
parameters={
    'model__n_estimators':[1,10],
    'model__min_samples_leaf':[1,5,10],
    'model__min_samples_split':[2,4,6]
    }

warnings.simplefilter(action='ignore', category=FutureWarning)
kfold=KFold(n_splits=5,shuffle=True)

pipe = Pipeline([("scaler",StandardScaler()),('model',RandomForestRegressor())]) 

# sorted(pipe.get_params().keys())

model=RandomizedSearchCV(pipe,parameters,cv=kfold,n_jobs=-1) 

model.fit(x_train , y_train)

print("최적의 매개변수=",model.best_params_)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_predict)
print("mse:",mse)

result=pipe.predict(test_data)
print(y_predict.shape)
print(y_predict)

a = np.arange(2800,3500)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp3/sample_submission1_3.csv", header = ["X","Y","M","V"], index = True, index_label="id" )

"""
mse: 114
"""