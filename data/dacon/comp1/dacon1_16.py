import numpy as np
import pandas as pd
# 1. 데이터
train=pd.read_csv('./data/dacon/comp1/train.csv',header=0,index_col=0)  #0행이 header, 0열이 index/ header와 index모두 존재
test=pd.read_csv('./data/dacon/comp1/test.csv',header=0, index_col=0)
submission=pd.read_csv('./data/dacon/comp1/sample_submission.csv',header=0,index_col=0)

# 이상치 처리

def outliers(data_out):
    out = []
    count = 0
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)
            count += len(out_col)

    if str(type(data_out))== str("<class 'pandas.core.frame.DataFrame'>"):
        i=0
        for col in data_out.columns:
            data = data_out[col].values
            quartile_1, quartile_3 = np.percentile(data,[25,75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print('dddd')
            print(out_col)
            print('aaa')
            print(out_col[0])
            data_out.iloc[out_col[0],i]=np.nan
            data = data[out_col]
            print(f"{col}의 이상치값: ", data)
            i+=1
    return data_out


train = outliers(train)
train = train.interpolate()
train = train.fillna(method='bfill')
print(train.isnull().sum)
test = outliers(test)
test = test.interpolate()
test = test.fillna(method='bfill')

np.save('./data/comp1_train.npy',arr=train)
np.save('./data/comp1_test.npy',arr=test)
train = np.load('./data/comp1_train.npy')
test = np.load('./data/comp1_test.npy')

from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

print(train.shape) #(10000,75)
print(test.shape)  #(10000,71) 



x = train[0:,:71]
y = train[0:,71:]

print(x.shape)  #(10000,71)
print(y.shape)  #(10000,4)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=66)

model = MultiOutputRegressor(LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, colsample_bytree=0.8))

model.fit(x_train,y_train)


score = model.score(x_test,y_test)

print("R2:",score)

# thresholds = np.sort(model.feature_importances_) # 오름차순 정렬(feature_importances정렬)
# print(thresholds)

# models=[]
# res = np.array([])
# for thresh in thresholds: 
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)  
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)

#     model2 = LGBMRegressor(n_estimators=500, learning_rate=0.1, n_jobs=-1)
#     model2.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','rmse'],
#                 eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
#                 early_stopping_rounds=20)
    
#     y_pred = model2.predict(select_x_test)
#     select_test = selection.transform(test)
    
#     score = r2_score(y_test,y_pred)
#     mae = mean_absolute_error(y_test,y_pred)
#     print("r2:",score)
#     print("mae:",mae)
   
print("r2:",score)
y_predict = model.predict(x_test)
mae = mean_absolute_error(y_test,y_predict)
print("mae:",mae)

result = model.predict(test)
a = np.arange(10000,20000)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission1_16.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )