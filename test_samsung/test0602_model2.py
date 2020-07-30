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
print("samsung:",samsung)
#samsung만 x, y분리해주면 된다
#hite는 x만 필요

#데이터 자르기
x_sam=samsung[:,0:5]
print(x_sam.shape) #(504,5)
y_sam=samsung[:,5]
print(y_sam.shape)#(504,)

x_sam=x_sam.reshape(504,5,1)


#hite는 x만 분리
x_hite=hite[5:510,0:5] 
print(x_hite.shape) #(504,5)
print(x_hite)

x_hite=x_hite.reshape(504,5,1)

#2. 모델 구성
input1=Input(shape=(5,1))
x1=LSTM(10)(input1)
x1=Dense(10)(x1)

input2=Input(shape=(5,1))
x2=LSTM(5)(input2)
x2=Dense(5)(x2)

merge=concatenate([x1,x2])

output=Dense(1)(merge)

model=Model(inputs=[input1,input2],outputs=output)

model.summary()

#3. 컴파일, 훈련
# 앙상블은 행이 맞아야 한다. 
model.compile(optimizer='adam',loss='mse')
model.fit([x_sam,x_hite],y_sam,epochs=5) 
