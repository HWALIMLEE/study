from xgboost import XGBClassifier, plot_importance , XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


x,y = load_breast_cancer(return_X_y=True)

print(x.shape) # (500,13)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)


parameters=[
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.5,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01],
    "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]}
]


model = GridSearchCV(XGBClassifier(),parameters,cv=5, n_jobs=-1) #n_jobs = -1 맨 뒤에
model.fit(x_train,y_train)

# 최적의 best값
print("===============================")
print(model.best_estimator_)
print("================================")
print(model.best_params_)
print("===============================")


score = model.score(x_test,y_test)
print("점수:",score)

# print(model.feature_importances_)  엑스지부스터에서만 먹힘

# plot_importance(model)
# plt.show()


#======================================결과==============================#
"""
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
================================
{'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
===============================
점수: 0.9736842105263158
"""
#===================================결과===================================#