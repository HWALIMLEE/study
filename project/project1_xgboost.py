import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

data = pd.read_csv("./project/data/data_company.csv",index_col=0, header=0, thousands=',')
test_data = pd.read_csv("./project/test_data/data_company_test.csv",index_col=0, thousands=',',header=None)

# 결측치 처리-보간법 이용
data = data.interpolate()
test_data = test_data.interpolate()
# 나머지 결측치-0으로 채워넣음
data = data.fillna(0)
test_data = test_data.fillna(0)

print(data)
print(test_data)

np.save("./project/data_company.npy",arr = data)
np.save("./project/data_company_test.npy",arr = test_data)
data = np.load("./project/data_company.npy", allow_pickle=True)
test_data = np.load("./project/data_company_test.npy",allow_pickle=True)

# integer변환
x = data[0:,0:-1].astype('int64')
y = data[0:,-1]
test_data = test_data.astype('int64')

# y분류 LabelEncoder로 숫자로 변경(0-건전, 1-부실)
encoder=LabelEncoder()
encoder.fit(y)
y_encode = encoder.transform(y)

print("test_data:",test_data)
print("x:",x)
print("y:",y_encode)

print("x.shape:",x.shape) #(150,135)
print("y.shape:",y.shape) #(150,)


#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y_encode,test_size=0.3,random_state=60)

print(x_train.shape)
print(x_test.shape)
print(y_train)

#2. 모델
model = XGBClassifier()

#3. 훈련, 평가예측
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("y_pred:",y_pred)

acc = model.score(x_test,y_test)
print("acc:",acc)  

#--------------------------------#
# acc : 0.978
#--------------------------------#

mse = mean_squared_error(y_test,y_pred)
print("mse:",mse)

#-------------------------------#
# mse : 0.02
#-------------------------------#

result=model.predict(test_data)
print("result:",result)

#-------------------------------#
# 임의의 기업 넣어서 test
#-------------------------------#
