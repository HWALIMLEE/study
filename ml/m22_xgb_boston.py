# 과적합 방지
# 1. 훈련 데이터를 늘린다. 
# 2. 피처수를 줄인다. 
# 3. regularization

from xgboost import XGBClassifier, plot_importance , XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x,y = load_boston(return_X_y=True)

print(x.shape) # (500,13)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)

#---------------------------------------------------------------------------------------------#
# 랜포 = 트리모델이 합쳐짐--->업그레이드 된 게 부스팅
# 트리모델 : 전처리 안해도 된다, 결측치 제거 안해도 된다
# 보간법 안해도된다(NaN자동으로 보간해주지만, 직접 보간하고 모델을 돌리는 게 더 나을 수도 있다.)
# 학습률(default = 0.01), 머신러닝, 딥러닝 모두 동일하게 쓰고 매우 중요하다
#---------------------------------------------------------------------------------------------#

# 파라미터는 이 정도만 알고 있으면 된다. 
n_estimators = 1000    # 나무개수
learning_rate = 0.01    # 제일 중요하다
colsample_bytree = 0.999  # 열샘플링- 컬럼에 대한 샘플링을 통해 각각의 다양성을 높인다./ 나무 필터링
colsample_bylabel = 0.999 # 열샘플링 - 컬럼에 대한 샘플링 / 라벨 필터링

max_depth = 3
n_jobs = -1          # 딥러닝일 경우만 제외하고 n_jobs = -1쓰기

#----------------------------------------------------------------------------------------------------#
#n_estimators = 1000     # 나무개수
# learning_rate = 0.01    # 제일 중요하다
# colsample_bytree = 0.99  # 열샘플링- 컬럼에 대한 샘플링을 통해 각각의 다양성을 높인다./ 나무 필터링
# colsample_bylabel = 0.99 # 열샘플링 - 컬럼에 대한 샘플링 / 라벨 필터링

# max_depth = 3
# n_jobs = -1          # 딥러닝일 경우만 제외하고 n_jobs = -1쓰기
# 0.9415922339014444
#--------------------------------------------------------------------------------------------------------#

model  = XGBRegressor(max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators, 
                        n_jobs = n_jobs, colsample_bylabel = colsample_bylabel,colsample_bytree = colsample_bytree)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("점수:",score)

"""
print(model.feature_importances_)
print("==========================")
print(model.best_estimators_)
print("==========================")
print(model.best_params_)
print("=========================")

"""
# 중요!!!!!!!!!
# 직접 함수 만드는 것도 가능(저번 시간에 했음)
plot_importance(model)
plt.show()