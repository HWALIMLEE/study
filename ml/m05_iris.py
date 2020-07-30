from sklearn.datasets import load_iris
# SVC, KNeighborsClassifier, KNeigoborsRegressor, LinearSVC,RandomForest
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

iris=load_iris()

x=iris.data
y=iris.target

 
#전처리
scaler=StandardScaler()
scaler.fit(x)
x_scaled=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8,shuffle=True)

#3. 훈련
print("-------------------RandomForestClassifier-----------------------------")
model=RandomForestClassifier() 
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)
print("y_preditc:",y_predict)
print("acc:",acc)                #0.96
print("score:",score)            #0.96

print("------------------------KNeighborsClassifier----------------------------")
model=KNeighborsClassifier()   

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_preditc:",y_predict) 
print("acc:",acc)                 #0.96
print("score:",score)             #0.96
#0.96/0.95-->KNeighborsRegressor 넣으면 score은 r2값 반환/ 원래 acc는 acc반환--> 값이 다른 이유

print("----------------------LinearSVC------------------------------")
model=LinearSVC()              

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_preditc:",y_predict)
print("acc:",acc)             #0.96
print("score:",score)         #0.96

print("---------------------SVC-------------------------")
model=SVC()                    
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_preditc:",y_predict)
print("acc:",acc)             #0.96
print("score:",score)         #0.96

print("---------------------RadomForestRegressor-----------------------")
model=RandomForestRegressor() 
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

# acc=accuracy_score(x_test,y_test)
r2=r2_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_preditc:",y_predict)
print("r2:",r2)              #0.96
print("score:",score)        #0.96
# print("acc:",acc)          #error 

print("---------------------KNeighborsRegressor---------------------")
model=KNeighborsRegressor()      
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_predict)
score=model.score(x_test,y_test)
# acc=accuracy_score(x_test,y_test)

print("y_preditc:",y_predict)
print("r2:",r2)                #0.95
print("score:",score)          #0.95
# print("acc:",acc)            #error

print("-----------------------------------------")

#결측치 제거
