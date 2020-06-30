import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/dacon/comp4/201901-202003.csv',header=0)
submission = pd.read_csv('./data/dacon/comp4/submission.csv',header=0, index_col=0)


def grap_year(data):
    data = str(data)
    return int(data[:4])

def grap_month(data):
    data = str(data)
    return int(data[4:])

# 날짜처리
data = data.fillna('')  #fillna함수도 굉장히 유용한다 NaN을 특정 값으로 대체하는 기능을 한다. 특정 텍스트라던지, 평균값이라던지... 굉장히 유효한 함수니까 필히 암기하자.

data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x))
data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))
data = data.drop(['REG_YYMM'],axis=1)

print(data)

# 데이터 정제
df = data.copy()
df = df.drop(['CARD_CCG_NM','HOM_CCG_NM'],axis=1)
columns = ['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM',"AGE",'SEX_CTGO_CD','FLC','year','month']
df = df.groupby(columns).sum().reset_index(drop=False)
print(df)

# 인코딩
dtypes = df.dtypes
encoders = {}
for column in df.columns:
    if str(dtypes[column])=='object':
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder

df_num = df.copy()
for column in encoders.keys():
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])


# 데이터의 타입 체크
df.info()

df['SEX_CTGO_CD'] = df['SEX_CTGO_CD'].astype(object)
df['FLC'] = df['FLC'].astype(object)

df.info()

# df['AMT'].value_counts().plot(kind='bar')
# plt.show()

category_feature = [col for col in df.columns if df[col].dtypes=='object']
print(category_feature)

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 명목형 변수의 분포 살펴보기
for col in category_feature:
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

train_num = df_num.sample(frac=1,random_state=0)
train_features = train_num.drop(['CSTMR_CNT','AMT','CNT'],axis=1)
print(train_features)
print("==============================================")
print(train_features.shape)
train_target = np.log1p(train_num['AMT'])

# log1p => 로그함수의 경우 x=0인 경우 y가 -무한대의 값을 가지게 됨
# 이럴 때 'Runtime Warning: divide by zero encountered in log'라는 경고 메시지 출력
# 이럴 때 사용하는 방법이 x+1 을 해줘서 0->1로 바꿔주는 것
# 이 역할을 np.log1p()가 해준다. (-inf -> 0)

parameters={
    'n_estimators':[10,100,1000],
    'max_depth':[6,8,10,-1],
    'colsample_bytree':[0.5,0.8,1],
    'learning_rate':[0.05,0.1,0.07],
    'num_leaves':[10,20,30],
}
kfold=5
model=RandomizedSearchCV(LGBMRegressor(),parameters,cv=kfold,n_jobs=-1) # 가장 좋은 것만 뽑아냄(시간이 훨씬 빠름/ 드롭아웃 생각하기)
model.fit(train_features, train_target)

# 예측 템플릿 만들기
# 고유값만 가지고 옴
CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()
STD_CLSS_NMs  = df_num['STD_CLSS_NM'].unique()
HOM_SIDO_NMs  = df_num['HOM_SIDO_NM'].unique()
AGEs          = df_num['AGE'].unique()
SEX_CTGO_CDs  = df_num['SEX_CTGO_CD'].unique()
FLCs          = df_num['FLC'].unique()
years         = [2020]
months        = [4, 7]

# 조합
temp = []
for CARD_SIDO_NM in CARD_SIDO_NMs:
    for STD_CLSS_NM in STD_CLSS_NMs:
        for HOM_SIDO_NM in HOM_SIDO_NMs:
            for AGE in AGEs:
                for SEX_CTGO_CD in SEX_CTGO_CDs:
                    for FLC in FLCs:
                        for year in years:
                            for month in months:
                                temp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])
temp = np.array(temp)
temp = pd.DataFrame(data=temp, columns=train_features.columns)
print(temp)
print(temp.shape)

pred = model.predict(temp)
pred = np.expm1(pred)
temp['AMT'] = np.round(pred,0)
temp['REG_YYMM'] = temp['year']*100+temp['month']
temp = temp[['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM','AMT']]
temp = temp.groupby(['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM']).sum().reset_index(drop=False)
# mae = mean_absolute_error(pred,train_target)
# print("mae:",mae)

#디코딩
temp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(temp['CARD_SIDO_NM'])
temp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(temp['STD_CLSS_NM'])

# 제출파일 만들기
submission = submission.drop(['AMT'],axis=1)
submission = submission.merge(temp, left_on=['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM'],right_on=['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM'],how='left')
submission.index.name='id'
submission.to_csv('./data/dacon/comp4/submission1.csv',encoding='utf-8-sig')
submission.head()
