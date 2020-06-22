from xgboost import XGBClassifier, plot_importance , XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x,y = load_breast_cancer(return_X_y=True)

print(x.shape) # (500,13)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)

n_estimators = 10000    # 나무개수
learning_rate = 0.01    # 제일 중요하다
colsample_bytree = 0.999  # 열샘플링- 컬럼에 대한 샘플링을 통해 각각의 다양성을 높인다./ 나무 필터링
colsample_bylabel = 0.999 # 열샘플링 - 컬럼에 대한 샘플링 / 라벨 필터링

max_depth = 5
n_jobs = -1          # 딥러닝일 경우만 제외하고 n_jobs = -1쓰기

model  = XGBClassifier(max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators, 
                        n_jobs = n_jobs, colsample_bylabel = colsample_bylabel,colsample_bytree = colsample_bytree)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("점수:",score)
