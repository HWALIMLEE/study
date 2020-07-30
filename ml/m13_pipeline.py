import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                                    random_state=43)
                                                    
#2. 모델
# model=SVC()
# svc_model=SVC()--->위와 동일

from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler #pipeline의 친구는 전처리

pipe = Pipeline([("scaler",MinMaxScaler()),('svm',SVC())])
pipe = make_pipeline(MinMaxScaler(),SVC()) #이름 따로 명시 안해줌
# 전처리와 모델 한번에 돌리는 것

pipe.fit(x_train , y_train)

print("acc:",pipe.score(x_test,y_test)) 

# wrapper 쓰고 gridSearch에 넣는다 --->model, parameter, cv
# pipeline과 gridSearch엮기