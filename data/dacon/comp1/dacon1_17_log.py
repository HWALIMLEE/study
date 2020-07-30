import numpy as np
import matplotlib.pyplot as plt
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
import scipy.signal
import scipy as sp
import pandas as pd

train=pd.read_csv('./data/dacon/comp1/train.csv',header=0,index_col=0)  #0행이 header, 0열이 index/ header와 index모두 존재
test=pd.read_csv('./data/dacon/comp1/test.csv',header=0, index_col=0)
submission=pd.read_csv('./data/dacon/comp1/sample_submission.csv',header=0,index_col=0)

train = train.interpolate()
test = test.interpolate()

train = train.fillna(method='bfill')
test = test.fillna(method='bfill')

np.save('./data/comp1_train.npy',arr=train)
np.save('./data/comp1_test.npy',arr=test)
train = np.load('./data/comp1_train.npy')
test = np.load('./data/comp1_test.npy')



print(train.shape) #(10000,75)
print(test.shape)  #(10000,71) 

test_rho=test[:,0:1]
test_fourier=test[:,1:]
x = train[0:,:71]
y = train[0:,71:]
x_rho = x[:,0:1]
x_fourier = x[:,1:71]


print(x.shape)  #(10000,71)
print(y.shape)  #(10000,4)
print(x_fourier.shape) #(10000,70)


# for col in x_fourier:
#     f, P = sp.signal.periodogram(col, 44100, nfft=2**12)
# plt.subplot(211)
# plt.plot(f,P)
# plt.xlim(100,1900)
# plt.title("선형 스케일")


# plt.subplot(212)
# plt.semilogy(f,P)
# plt.xlim(100,1900)
# plt.ylim(1e-5,1e-1)
# plt.title("로그 스케일")

# plt.tight_layout()
# plt.show()

if x_fourier.all() >= 0:
    x_log = np.log(x_fourier)
    np.nan_to_num(x_log, copy=False)
    
else:
    x_fourier_abs = abs(x_fourier)
    x_log_plus = np.log(x_fourier_abs)
    np.nan_to_num(x_log_plus, copy=False)
    x_log = - x_log_plus
    

x_log_fin = np.concatenate((x_rho,x_log),axis=1)
print(x_log_fin)

if test_fourier.all() >= 0:
    test_log = np.log(test_fourier)
    np.nan_to_num(test_log, copy=False)
    
else:
    test_fourier_abs = abs(test_fourier)
    test_log_plus = np.log(test_fourier_abs)
    np.nan_to_num(test_log_plus, copy=False)
    test_log = - test_log_plus

test_fin = np.concatenate((test_rho,test_log),axis=1)   

x_log_fin = np.concatenate((x_rho,x_log),axis=1)
x_train,x_test,y_train,y_test = train_test_split(x_log_fin,y,test_size=0.1,random_state=66)

model = MultiOutputRegressor(LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, colsample_bytree=0.8))

model.fit(x_train,y_train)

score = model.score(x_test,y_test)

print("R2:",score)

y_predict = model.predict(x_test)
mae = mean_absolute_error(y_test,y_predict)
print("mae:",mae)

result = model.predict(test_fin)
a = np.arange(10000,20000)
#np.arange--수열 만들때
submission = result
submission = pd.DataFrame(submission, a)
submission.to_csv("./data/dacon/comp1/sample_submission1_log.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
