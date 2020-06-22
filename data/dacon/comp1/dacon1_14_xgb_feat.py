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
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error

print(train.shape) #(10000,75)
print(test.shape)  #(10000,71) 


x = train[0:,:71]
y = train[0:,71:]

print(x.shape)  #(10000,71)
print(y.shape)  #(10000,4)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=66)

pca=PCA(n_components=1, whiten=True,random_state=60).fit(y_train)
y_train_pca=pca.transform(y_train)
y_test_pca=pca.transform(y_test)

print(y_train_pca.shape) #(9000,1)
print(y_test_pca.shape)  #(1000,1)

model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, colsample_bytree=0.7,colsample_bylevel=0.7)

model.fit(x_train,y_train_pca)

score = model.score(x_test,y_test_pca)

print("R2:",score)

thresholds = np.sort(model.feature_importances_) # 오름차순 정렬(feature_importances정렬)
print(thresholds)

for thresh in thresholds: # 컬럼수만큼 돈다(최소한 13번)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  
    
    select_x_train = selection.transform(x_train)

    # GridSearch 넣고, submit하기
    
    parameters=[
        {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01],
        "max_depth":[4,5,6]},
        {"n_estimators":[10,100,100], "learning_rate":[0.1,0.001,0.01],
        "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
        {"n_estimators":[10,100,1000], "learning_rate":[0.1,0.001,0.01],
        "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
        "colsample_bylevel":[0.6,0.7,0.9]}
        ]

    selection_model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1)
    selection_model.fit(select_x_train,y_train_pca)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    select_test = selection.transform(test)
    
    score = r2_score(y_test_pca,y_pred)
    mae = mean_absolute_error(y_test_pca,y_pred)
    

    # print("R2:",score)
    print(selection_model.best_params_)
    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
    print('Thresh=%.3f, n=%d, mae: %.2f' %(thresh, select_x_train.shape[1], mae))

    result = selection_model.predict(select_test)
    result = result.reshape(10000,1)
    result_transform = pca.inverse_transform(result)
    a = np.arange(10000,20000)
    #np.arange--수열 만들때
    submission = result_transform
    submission = pd.DataFrame(submission, a)
    submission.to_csv("./data/dacon/comp1/sample_submission1_14.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
