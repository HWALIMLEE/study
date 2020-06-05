#LSTM두개 구현
#삼성전자
#hite는 lstm은 시계열 모델 아니다. 왜냐면 거래량 이런것이 있기 때문
#samsung은 lstm가능
#pca로 hite축소(n*1로 줄이기)
#삼성(n*1)=하이트(n*1)
import numpy as np
from keras.models import Model,Input
from keras.layers import Dense, LSTM, Dropout
from keras.layers.merge import Concatenate, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA


def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)] #행 구성
        aaa.append([item for item in subset]) #subset에 있는 아이템을 반환
    print(type(aaa))
    return np.array(aaa)

size=6

# 1.데이터
# npy불러오기
samsung=np.load('./data/samsung.npy',allow_pickle=True)
hite=np.load('./data/hite.npy',allow_pickle=True)


print("samsung.shape:",samsung.shape) #shape=(509,1)
print("hite.shape:",hite.shape)

samsung=samsung.reshape(samsung.shape[0],) #shape=(509,) 
samsung=(split_x(samsung,size))
print("samsung.shape:",samsung.shape)
#samsung만 x, y분리해주면 된다
#hite는 x만 필요

#데이터 자르기
x_sam=samsung[:,0:5]
print(x_sam.shape) #(504,5)
y_sam=samsung[:,5]
print(y_sam.shape)#(504,)

x_sam=x_sam.reshape(504,5,1)

x_hite=hite[0:,0:4]
print("x_hite:",x_hite)

x_hite=StandardScaler().fit_transform(x_hite)

# pca=PCA(n_components=1) #주성분 개수
# trans_x_hite=pca.fit_transform(x_hite)

print("trans.shpe:",x_hite.shape)

x_hite=(split_x(x_hite,size))

print(x_hite.shape)

x_hite=x_hite[:,0:5]

print(x_hite.shape) #(504,5,1)

x_sam_train,x_sam_test,y_sam_train,y_sam_test, x_hite_train,x_hite_test=train_test_split(x_sam,y_sam,x_hite,test_size=0.2,random_state=60)

#2. 모델 구성
input1=Input(shape=(5,1)) # 변수명은 소문자(암묵적약속)
dense1_1=LSTM(30,activation='relu',name='A1')(input1) #input명시해주어야 함
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_2=Dense(50,activation='relu',name='A2')(dense1_1)
dense1_3=Dense(40,activation='relu',name='A3')(dense1_2)
dense1_4=Dense(30,activation='relu',name='A4')(dense1_3)

#두번쨰 모델(2)
input2=Input(shape=(5,4)) 
dense2_1=LSTM(30,activation='relu',name='B1')(input2) 
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense1_2=Dense(40,activation='relu',name='A2')(dense1_1)
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_2=Dense(40,activation='relu',name='B2')(dense2_1)
dense2_3=Dense(30,activation='relu',name='B3')(dense2_2)
dense2_4=Dense(20,activation='relu',name='B4')(dense2_3)

# 엮어주는 기능(첫번째 모델과 두번째 모델)(3) #concatenate-사슬 같이 잇다, 단순병합
from keras.layers.merge import concatenate
merge1=concatenate([dense1_4,dense2_4],name='merge1') #두 개 이상은 항상 리스트('[]')

# 또 레이어 연결(3)
middle1=Dense(10)(merge1)
middle1=Dense(15)(middle1)
middle1=Dense(15)(middle1)
middle1=Dense(10)(middle1)

# 엮은 거 다시 풀어준다(output도 2개 나와야하니까) ---분리(4)
# input=middle1(상단 레이어의 이름)
output1=Dense(10,name='o1')(middle1)
output1_2=Dense(70,name='o1_2')(output1)
output1_3=Dense(30,name='o1_3')(output1_2) 
output1_4=Dense(30,name='o1_4')(output1_3) 
output1_5=Dense(1,name='o1_5')(output1_4) 


#함수형 지정(제일 하단에 명시함)
model=Model(inputs=[input1,input2], outputs=output1_5) # 범위 명시 #함수형은 마지막에 선언 #두 개 이상은 리스트

model.summary()

#3. 컴파일, 훈련
# 앙상블은 행이 맞아야 한다. 
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=10, mode='aut')
model.fit([x_sam_train,x_hite_train],y_sam_train,epochs=100,batch_size=1,callbacks=[early_stopping])

loss,acc=model.evaluate([x_sam_test,x_hite_test],y_sam_test,batch_size=1)
y_predict=model.predict([x_sam_test,x_hite_test])
print("y_predict:",y_predict)

for i in range(5):
    print("종가:",y_sam_test[i],'/예측가:',y_predict[i])
