import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel


#불러오기
train_data=np.load('./data/comp3_train_features.npy')
train_target=np.load('./data/comp3_train_target.npy')
test_data=np.load('./data/comp3_test_features.npy')

train_data = train_data[0:,1:] # 시간 제외
test_data = test_data[0:,1:]   # 시간 제외

print(train_data.shape)  #(1050000,4)
print(train_target.shape) #(2800,4)
print(test_data.shape)    #(262500,4)
"""
x_train,x_test,y_train,y_test = train_test_split(train_data,train_target,test_size=0.2)

x_train=x_train.reshape(8960,375)
x_test=x_test.reshape(2240,375)
y_train=y_train.reshape(8960,)
y_test=y_test.reshape(2240,)

print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
print("y_train.shape:",y_train.shape)
print("y_test.shape:",y_test.shape)
print("test.shape:",test_data.shape)


model= XGBRegressor()

model.fit(x_train,y_train)


y_predict=model.predict(x_test)

test_data=test_data.reshape(2800,375)
result=model.predict(test_data)

print(y_predict.shape)
print(y_predict)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_predict)
print("mse:",mse)

score=model.score(x_test,y_test)
print("score:",score)


a = np.arange(2800,3500)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp3/sample_submission1_xgb.csv", header = ["X","Y","M","V"], index = True, index_label="id" )
"""