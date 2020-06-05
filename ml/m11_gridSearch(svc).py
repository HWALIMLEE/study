#gridSearch ==> 하나도 빼지 않고 모두 조사하는 것,하이퍼 파라미터 다 찾는거
#모든 것이 다 있는다고 해서 좋은 성능을 내는 것은 아니다
#randomSearch ==> gridSearch와 같이 쓰면 good
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV #여기서 cv는 cross_validation
from sklearn.metrics import accuracy_score
import warnings 
from sklearn.svm import SVC

#1. 데이터
iris=pd.read_csv('./data/csv/iris.csv',header=0)

x=iris.iloc[:,0:4] #0,1,2,3
y=iris.iloc[:,4]

#numpy일 때는 그냥 슬라이싱 해주어도 된다

print("x:",x)
print("y:",y)
warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다

kfold=KFold(n_splits=5,shuffle=True) #현재 데이터 5 등분으로 나누겠다, 그렇다면 20%는 테스트, 80%훈련, 이걸 다섯번 반복

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44) 

# SVC모델에서 제공되는 parameter
parameters=[
    {"C":[1,10,100,1000],"kernel":["linear"]}, #에포, 활성화함수의 냄새가 난다
    {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]}, 
    {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}

]

kfold=KFold(n_splits=5,shuffle=True)
model=GridSearchCV(SVC(),parameters,cv=kfold) #몇개로 validation할 것인가 (kfold만큼=5(위에서 5라고 지정))
# train 다섯조각 내서 20%만큼 검증 셋으로 주겠다
# 20%의 테스트는 완전 별도
# train의 20%를 돌려가며 검증하겠다
# 완벽하게 train,test,validation분리 됨
# train에서만 20%뽑아내는 것
# test는 안 건들여짐

#1. 진짜모델
#2. 파라미터
#3. cv

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("y_pred:",y_pred)

print("최종 정답률:",accuracy_score(y_test,y_pred))
print("최적의 매개변수:",model.best_estimator_)
"""
최적의 매개변수: SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,   
  shrinking=True, tol=0.001, verbose=False)
  
  가장 좋은 파라미터 값 선택해줌
"""

"""
무엇이 1.0이 나왔는 지 알 수 없다
어떤 매개변수가 최적?
어떻게 찾을까?
"""


