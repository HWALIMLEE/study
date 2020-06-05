# 이렇듯 고정된 train set과 test set으로 평가를 하고,
# 반복적으로 모델을 튜닝하다보면 test set에만 과적합되어버리는 결과가 생긴다.
# 이를 해결하고자 하는 것이 바로 교차 검증(cross validation)이다.
# 설명: https://m.blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221599517834&proxyReferer=https:%2F%2Fwww.google.com%2F
# 그러나 iteration횟수가 많기 때문에 모델 훈련/평가 시간이 오래 걸린다
# 홀드아웃 방법, k-겹 교차 검증, 리브 p-아웃 교차 검증, 리브-원-아웃 교차 검증, 계층별 k-겹 교차 검증
# 즉 과적합 방지하기 위함

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #26개 모델 한방에 돌려버림
import warnings 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score #KFold에서 쓰는 score

warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다
#1. 데이터
iris=pd.read_csv('./data/csv/iris.csv',header=0)

x=iris.iloc[:,0:4] #0,1,2,3
y=iris.iloc[:,4]

#numpy일 때는 그냥 슬라이싱 해주어도 된다

print("x:",x)
print("y:",y)
warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다

kfold=KFold(n_splits=5,shuffle=True) #현재 데이터 5 등분으로 나누겠다, 그렇다면 20%는 테스트, 80%훈련, 이걸 다섯번 반복

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44) train_test_split 필요 없음

#3. 모델
warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다
allAlgorithms = all_estimators(type_filter='classifier') #분류모델 싹 다 가져옴-->멋진 아이임

for (name,algorithm) in allAlgorithms: #name과 algorithm이 반환값
    model=algorithm()
    
    scores=cross_val_score(model,x,y,cv=kfold) 
    #통으로 넣고 kfold=5기 때문에 5개로 자르고, 훈련시켜서 score내주겠다

    print(name,"의 정답률=")
    print(scores) 
    #여기서 score는 accuracy
    #각 모델별로 5번씩 돌린 accuracy 출력됨

import sklearn
print(sklearn.__version__)

#sklearn버전 낮추면 all_estimators 정상 작동


#커밋수정
