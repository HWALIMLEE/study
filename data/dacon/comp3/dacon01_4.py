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

train_features=pd.read_csv("./data/dacon/comp3/train_features.csv",header=0)
train_target=pd.read_csv("./data/dacon/comp3/train_target.csv",header=0)
test_features=pd.read_csv("./data/dacon/comp3/test_features.csv",header=0)

print(train_features)

def preprocessing_KAERI(data):
    """
    data: train_features.csv or test_features.csv

    return: RandomForest 모델 입력용 데이터
    """

# 충돌체 별로 0.000116초 까지의 가속도 데이터만 활용해보기
_data = train_features.groupby('id').head(375)

# string형태로 변환
_data['Time'] = _data['Time'].astype('str')

# RandomForest 모델에 입력할 수 있는 1차원 형태로 가속도 데이터 변환
_data = _data.pivot_table(index="id",columns="Time",values=["S1","S2","S3","S4"])