import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential


tit=pd.read_csv("C:\\Users\\hwalim\\kaggle\\train.csv")
tit_test=pd.read_csv("C:\\Users\\hwalim\\kaggle\\test.csv")
print(tit.shape) #(891,12)
print("------------------")
print(tit.isnull().sum())

tit.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
print("---------------------")
print(tit['Cabin'].value_counts())
print(tit['Embarked'].value_counts())
print("-------------------------")
tit['Age'].fillna(tit['Age'].mean(),inplace=True)
tit['Cabin'].fillna('n',inplace=True)
tit['Embarked'].fillna('S',inplace=True)
print(tit.isnull().sum())

tit['Cabin']=tit['Cabin'].str[:1]
tit['Cabin'].head()

tit[tit['Age']<=0]

def age(x):
    Age=''
    if x <= 12: 
          Age='Child'
    elif x <= 19: 
        Age='Teen'
    elif x <= 30: 
        Age='Young_adult'
    elif x <= 60: 
        Age='Adult'
    else: 
        Age='Old'
        
    return Age
print("-------------------")
tit['Age']=tit['Age'].apply(lambda x:age(x))
print(tit['Age'].isnull().any())
print("-------------------")
#encoding
from sklearn.preprocessing import LabelEncoder
def encoding(x):
    for i in ['Sex','Age','Cabin','Embarked']:
        x[i]=LabelEncoder().fit_transform(x[i])
    return x

tit=encoding(tit)
print(tit.head())

#OneHotEncoding(pd.get_dummies)
tit=pd.get_dummies(tit,columns=['Pclass','Sex','Age','Embarked'])
tit.head()
print("----------------------")
#레이블 데이터 분리
y_train=tit['Survived']
x_train=tit.drop('Survived',axis=1)
print(x_train.shape)
print(y_train.shape)
print(tit_test.shape)

model=Sequential()
model.add(Dense(10,input_dim=17))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(x_train,y_train,epochs=10,batch_size=1)


results=model.predict(tit_test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name="label")

submission=pd.concat([pd.Series(range(1,419),name="Survied"),results],axis=1)
submission.to_csv("D:\\STUDY\\kaggle\\cnn_titanic_dataset.csv",index=False)
