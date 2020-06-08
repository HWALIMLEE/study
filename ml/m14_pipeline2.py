#RandomizedSearchCV + Pipeline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 1. 데이터
iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                                    random_state=43)

# 그리드/랜덤 서치에서 사용할 매개 변수
# 돌릴때마다 최적 매개변수 값이 바뀐다
parameters=[
    {"svm__C":[1,10,100,1000], "svm__kernel":['linear']},                             #4가지 
    {"svm__C":[1,10,100,1000], "svm__kernel":['rbf'],'svm__gamma':[0.001,0.0001]},    #8가지
    {"svm__C":[1,10,100,1000], "svm__kernel":['sigmoid'],'svm__gamma':[0.001,0.0001]} #8가지
    #총 20가지 가능한 parameter

]

# 위에꺼랑 동일하다/한번에 넣어주기
"""
parameters=[
    {"svm__C":[1,10,100,1000], "svm__kernel":['linear','rbf','sigmoid'],'svm__gamma':[0.001,0.0001]}
]
"""


# 2. 모델
# model=SVC()
# svc_model=SVC()--->위와 동일

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler #pipeline의 친구는 전처리

pipe = Pipeline([("scaler",MinMaxScaler()),('svm',SVC())]) 
# 전처리와 모델 한번에 돌리는 것

model = RandomizedSearchCV(pipe, parameters, cv=5) #pipe가 모델

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
acc=model.score(x_test,y_test)
print("최적의 매개변수=",model.best_estimator_)
print("acc:",acc)

# wrapper 쓰고 gridSearch에 넣는다 --->model, parameter, cv
# pipeline과 gridSearch엮기

