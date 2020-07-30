from sklearn.datasets import load_breast_cancer
# SVC, KNeighborsClassifier, KNeigoborsRegressor, LinearSVC,RandomForest
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

cancer=load_breast_cancer()

x=cancer.data
y=cancer.target

scaler=StandardScaler()

scaler.fit(x)
x_scaled=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8)

print("-------------------RandomForestClassifier-----------------------------")
model=RandomForestClassifier() 
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)
print("y_predict:",y_predict)
print("acc:",acc)                #0.98
print("score:",score)            #0.98

print("------------------------KNeighborsClassifier----------------------------")
model=KNeighborsClassifier()   

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_predict:",y_predict) 
print("acc:",acc)                 #0.97
print("score:",score)             #0.97

print("----------------------LinearSVC------------------------------")
model=LinearSVC()              

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_predict:",y_predict)
print("acc:",acc)             #0.95
print("score:",score)         #0.95

print("---------------------SVC-------------------------")
model=SVC()                    
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

acc=accuracy_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_predict:",y_predict)
print("acc:",acc)             #0.99
print("score:",score)         #0.99

print("---------------------RadomForestRegressor-----------------------")
model=RandomForestRegressor() 
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_predict:",y_predict)
print("r2:",r2)              #0.87
print("score:",score)        #0.87
 
print("---------------------KNeighborsRegressor---------------------")
model=KNeighborsRegressor()      
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_predict)
score=model.score(x_test,y_test)

print("y_predict:",y_predict)
print("r2:",r2)                #0.90
print("score:",score)          #0.90

print("-----------------------------------------")