#보스턴 모델링 하시오
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #26개 모델 한방에 돌려버림
import warnings 
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다

boston=pd.read_csv('./data/csv/boston_house_prices.csv',header=1)

x=boston.iloc[:,0:13] #0,1,2,3
y=boston.iloc[:,13]

#numpy일 때는 그냥 슬라이싱 해주어도 된다

print("x:",x)
print("y:",y)
warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

#3. 모델
warnings.filterwarnings('ignore') #warnings이라는 에러 그냥 넘어가겠다
allAlgorithms = all_estimators(type_filter='regressor') #분류모델 싹 다 가져옴-->멋진 아이임


for (name,algorithm) in allAlgorithms: #name과 algorithm이 반환값
    model=algorithm()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score=model.score(x_test,y_test)
    print(name,"의 정답률",score)
    
import sklearn
print(sklearn.__version__)

#sklearn버전 낮추면 all_estimators 정상 작동


#커밋수정