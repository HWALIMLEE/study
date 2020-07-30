import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

data = pd.read_csv("./project/data/data_company.csv",index_col=0, header=0, thousands=',')
# data = data.interpolate()
test_data = pd.read_csv("./project/test_data/data_company_test.csv",index_col=0, thousands=',',header=None)
# test_data = test_data.interpolate()
data = data.fillna(0)
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

# scaler1=StandardScaler()
# scaler1.fit(x_train)
# x_train_scaled = scaler1.fit_transform(x_train)

print("x_train_scaled:",x_train)

#2. 모델
model = RandomForestClassifier()

#3. 훈련, 평가예측
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("y_pred:",y_pred)

acc = model.score(x_test,y_test)
print("acc:",acc)

result=model.predict(test_data)

print("result:",result)

#1은부실, 0은 건전

#-----------------------------#
# acc:0.9111
# interpolate 안쓰니 0.78
#-----------------------------#