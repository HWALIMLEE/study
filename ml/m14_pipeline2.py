# RandomizedSearchCV + Pipeline
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
print(x_train.shape)
print(x_test.shape)

# 그리드/랜덤 서치에서 사용할 매개 변수
# 돌릴때마다 최적 매개변수 값이 바뀐다
# 그리드 서치, 랜덤서치 단독으로 할때는 모델명 언급안해줘도 된다
parameters=[
    {"C":[1,10,100,1000], "kernel":['linear']},                             #4가지 
    {"C":[1,10,100,1000], "kernel":['rbf'],'gamma':[0.001,0.0001]},    #8가지
    {"C":[1,10,100,1000], "kernel":['sigmoid'],'gamma':[0.001,0.0001]} #8가지
    #총 20가지 가능한 parameter

]
# 파이프라인, 그리드 서치나 랜덤서치 엮게 될 때는 모델명(밑에서 우리가 임의로 지정한)을 파라미터 앞에 언급해줘야 한다./ 짝대기 두개
# 파이프라인은 무조건 변수명 언급
# 버전 상관 없음
parameters=[
    {"svm__C":[1,10,100,1000], "svm__kernel":['linear']},                             #4가지 
    {"svm__C":[1,10,100,1000], "svm__kernel":['rbf'],'svm__gamma':[0.001,0.0001]},    #8가지
    {"svm__C":[1,10,100,1000], "svm__kernel":['sigmoid'],'svm__gamma':[0.001,0.0001]} #8가지
    #총 20가지 가능한 parameter

]

# make_pipeline ---> 모델명 앞에 언급해줄 것, 소문자로 써야 한다. 
parameters=[
    {"svc__C":[1,10,100,1000], "svc__kernel":['linear']},                             #4가지 
    {"svc__C":[1,10,100,1000], "svc__kernel":['rbf'],'svc__gamma':[0.001,0.0001]},    #8가지
    {"svc__C":[1,10,100,1000], "svc__kernel":['sigmoid'],'svc__gamma':[0.001,0.0001]} #8가지
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

# pipe = Pipeline([("scaler",MinMaxScaler()),('svm',SVC())]) 

# make_pipeline 역시 이름 명시 해주어야 한다.--->SVC 쓰기 때문에 모델명 svc로 앞에 써주면 된다
pipe = make_pipeline(MinMaxScaler(),SVC())
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
