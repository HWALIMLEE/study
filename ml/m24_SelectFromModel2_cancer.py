from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=66)


parameters=[
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators":[10,100,100], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[10,100,1000], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]}
    ]

model = GridSearchCV(XGBClassifier(), parameters, cv=5, n_jobs=-1)    # 결측치, 전처리 안해도 돼
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("R2:",score)

thresholds = np.sort(model.best_estimator_.feature_importances_) # 오름차순 정렬(feature_importances정렬)
print(thresholds)

for thresh in thresholds: # 컬럼수만큼 돈다(최소한 13번)
    selection = SelectFromModel(model.best_estimator_,threshold=thresh,prefit=True)  
    
    select_x_train = selection.transform(x_train)

    print(select_x_train.shape) #컬럼수가 한개씩 줄어든다.중요하지 않은 순서대로 지우고 있음(feature_importances_순서대로)
    #================================#
    """
    (404, 13)
    (404, 12)
    (404, 11)
    (404, 10)
    (404, 9)
    (404, 8)
    (404, 7)
    (404, 6)
    (404, 5)
    (404, 4)
    (404, 3)
    (404, 2)
    (404, 1)
    """
    #===================================#

    # GridSearch 넣고, submit하기

    parameters=[
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators":[10,100,100], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[10,100,1000], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]}
    ]

    selection_model = GridSearchCV(XGBRegressor(),parameters,cv=5,n_jobs=-1)
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_pred)
    # print("R2:",score)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))



    # plot_importances써서 어떤 변수명인지 미리 확인해둘 것


    # Thresh=feature_importance