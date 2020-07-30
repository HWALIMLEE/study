import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sklearn
import matplotlib
import tqdm

#confusion_matrix: 학습된 모델이 얼마나 훌륭한지 판단하는 성능 지표

print('tqdm',tqdm.__version__)
print('lightgbm',lgb.__version__)

train = pd.read_csv('./data/dacon/comp3/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv')
test = pd.read_csv('./data/dacon/comp3/test_features.csv')
submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')


