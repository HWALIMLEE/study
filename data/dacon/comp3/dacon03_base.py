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

# 결측값 없음
print(train.isnull().sum())
print(train_target.isnull().sum())
print(test.isnull().sum())

# 분류모델인 것 같음
print(train_target.head())
print(train_target['X'].value_counts())
print(train_target['Y'].value_counts())
print(train_target['M'].value_counts())
print(train_target['V'].value_counts())


print(train['id'].nunique()) # 일정한 간격으로 375번 측정된 데이터가 2800개 있음
"""
# 각 센서에서 측정된 가속도(S1, S2, S3, S4)
def accerlation_show(idx):
    f,axes = plt.subplots(4,1)
    plt.figure(figsize=(3,3))
    f.tight_layout()
    plt.subplots_adjust(bottom=-0.4)
    for i in range(1,5):
        axes[i-1].plot(train[train['id']==idx]['S'+str(i)].values)
        axes[i-1].set_title('S'+str(i))
        axes[i-1].set_xlabel('time')
    plt.show()
accerlation_show(1)

M_175_id = train_target[train_target['M']==25]['id'].values # 질량이 가벼운 데이터는 위아래로 흔들리는 파동이 잦다
accerlation_show(M_175_id[100]) 


plt.figure(figsize=(10,10))
sns.scatterplot(train_target['X'].values,train_target['Y'].values)
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.show()
"""
# 아이디 별로 가속도 4개 일차원 배열로 변경
# 일차원 배열이어야 나중에 합칠 수 있다. 
from tqdm.notebook import tqdm
train_=[]
for ID in tqdm(train['id'].unique()):
    tmp_df = train[train['id']==ID]
    tmp_X = []
    tmp_X.append(tmp_df['S1'].values)
    tmp_X.append(tmp_df['S2'].values)
    tmp_X.append(tmp_df['S3'].values)
    tmp_X.append(tmp_df['S4'].values)
    train_.append(tmp_X)
train_=np.array(train_).reshape(train['id'].nunique(),-1)
print(train_.shape) #(2800,1500) #1500은 375*4
print(len(train_)) #(2800)

test_=[]
for ID in tqdm(test['id'].unique()):
    tmp_df = test[test['id']==ID]
    tmp_X = []
    tmp_X.append(tmp_df['S1'].values)
    tmp_X.append(tmp_df['S2'].values)
    tmp_X.append(tmp_df['S3'].values)
    tmp_X.append(tmp_df['S4'].values)
    test_.append(tmp_X)
test_=np.array(test_).reshape(test['id'].nunique(),-1)
print(len(test_)) # 700

# train과 test의 분포를 같게 만들어준다. 1:1 비율로
# np.random.choice
# 이미 있는 데이터 집합에서 일부를 무작위로 선택하는 것을 샘플링(sampling)이라고 한다. 
# sampling에서는 choice명령을 사용한다. 
np.random.seed(42)
choice_idxs = np.random.choice(range(len(train_)),len(test_),replace=False) # 2800개 중에서 700개만 추출(비복원)
X_train_test = np.concatenate((train_[choice_idxs],test_))
y_train_test = np.array([0 for _ in range(len(choice_idxs))]+[1 for _ in range(test['id'].nunique())])
print(X_train_test.shape) # 1400,1500
print(y_train_test.shape) # 1400,1500

# 층화추출을 통해 데이터 분리를 할 때 분포가 비슷하도록 만들어줌.
X_train,X_test,y_train,y_test = train_test_split(X_train_test,y_train_test,test_size=0.2,random_state=42)

print(X_train.shape) #(1120,1500)
print(X_test.shape) #(280,1500)
print(y_train.shape) #(1120,)
print(y_test.shape) #(280,)

params ={
    'objective': 'binary','metrics':'auc'
}
train_set = lgb.Dataset(X_train,y_train)
test_set = lgb.Dataset(X_test,y_test)
lgb_model = lgb.train(params,train_set,1000,early_stopping_rounds=200,valid_sets=[train_set,test_set],verbose_eval=50)

pred = lgb_model.predict(X_test)
pred = np.where(pred>=0.5,1,0)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# a = np.arange(2800,3500)
# #np.arange--수열 만들때
# submission = result
# submission = pd.DataFrame(submission, a)
# submission.to_csv("./data/dacon/comp3/sample_submission1.csv", header = ["X","Y","M","V"], index = True, index_label="id" )
