import pandas as pd
import matplotlib.pyplot as plt

#와인 데이터 읽기
wine=pd.read_csv('./data/csv/winequality-white.csv',sep=';',header=0)

x=wine.drop('quality',axis=1)  #qulaity만 제거한다. #외워두기
y=wine['quality']

print(x.shape)
print(y.shape)

#y레이블 축소
newlist=[]
for i in y:
    if i<=4:
        newlist+=[0]
    elif i<=7:
        newlist+=[1]
    else:
        newlist+=[2]

#wine의 quality를 세가지로 줄임
#아주 좋음, 좋음, 보통

y=newlist
print("newlist:",newlist)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#3. 모델
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

model.fit(x_train,y_train)

acc=model.score(x_test,y_test)

y_pred=model.predict(x_test)
print("정답률:",acc)

acc_score=accuracy_score(y_test,y_pred)
print("acc_score:",acc_score)

#커밋수정