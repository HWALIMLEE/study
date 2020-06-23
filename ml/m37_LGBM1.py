# 200622_25
# boston / model save


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
from lightgbm import LGBMRegressor

### 데이터 ###
x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = LGBMRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9313126937746082 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)


models = []
res = np.array([])

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    model2 = LGBMRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    model2.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','rmse'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = model2.predict(select_x_test)
    score = r2_score(y_test, y_pred)
    shape = select_x_train.shape
    models.append(model2)           # 모델을 배열에 저장
    print(thresh, score)
    res = np.append(res, score)     # 결과값을 전부 배열에 저장
print(res.shape)
best_idx = res.argmax()             # 결과값 최대값의 index 저장
score = res[best_idx]               # 위 인덱스 기반으로 점수 호출
total_col = x_train.shape[1] - best_idx # 전체 컬럼 계산
model = models[best_idx]
import pickle # 파이썬 제공 라이브러리
pickle.dump(model, open(f"./model/xgb_save/boston--{score}--{total_col}.pickle.dat", "wb"))    # wb 라는 형식으로 저장을 하겠다
print("저장됐다.")
# models[best_idx].save_model(f'./model/xgb_save/boston--{score}--{total_col}--.model')    # 인덱스 기반으로 모델 저장

