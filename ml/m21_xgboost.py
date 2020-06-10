from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# xgboost는 한번씩 꼭 넣기를 추천함
# 아직까지 우승 모델

cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(
    cancer.data, cancer.target, train_size=0.9, random_state=42)

model = XGBClassifier()

# max_features : 기본값 쓸 것
# n_estimators : 클수록 좋다, 단점은 메모리 많이 차지, 기본값 100
# n_jobs = -1; 병렬 처리, GPU같이 돌릴 때는 n_jobs = 1 권유


model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc:",acc)

print(model.feature_importances_) #feature_importances_ //컬럼 30개 중에 8번째 컬럼이 accuracy에 영향을 많이 끼친다
# 가중치 줄 때 유리하겠군
# 혹은 PCA쓸때

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features=cancer.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plt.subplots(figsize=(15,6))
plot_feature_importances_cancer(model)
plt.show()