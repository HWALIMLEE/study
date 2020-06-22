import numpy as np
import pandas as pd
# 1. 데이터
train=np.load('./data/comp1_train.npy')
test=np.load('./data/comp1_test.npy')

from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, train_test_split,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print(train.shape) #(10000,75)
print(test.shape)  #(10000,71) 


x = train[0:,:71]
y = train[0:,71:]

print(x.shape)  #(10000,71)
print(y.shape)  #(10000,4)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=66)

# y주성분 분석
pca = PCA(n_components=1, whiten=True,random_state=60).fit(y_train)
y_train_pca=pca.transform(y_train)
y_test_pca=pca.transform(y_test)

# # x주성분 분석
pca2 = PCA(n_components=53, whiten=True, random_state=60).fit(x_train)
x_train_pca = pca2.transform(x_train)
x_test_pca = pca2.transform(x_test)
test_pca = pca2.transform(test)

print(y_train_pca.shape) #(9000,1)
print(y_test_pca.shape)  #(1000,1)

model = XGBRegressor(n_estimators=1000,max_depth=6,learning_rate=0.01,colsample_bytree=0.6)

model.fit(x_train_pca,y_train_pca)

score = model.score(x_test_pca,y_test_pca)

print("r2:",score)

y_pred = model.predict(x_test_pca)

mae = mean_absolute_error(y_pred,y_test_pca)

print("mae:",mae)

result = model.predict(test_pca)
result = result.reshape(10000,1)
result_transform = pca.inverse_transform(result)


a = np.arange(10000,20000)
#np.arange--수열 만들때
submission = result_transform
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission1_13.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
