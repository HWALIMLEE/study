# 모델 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1. 데이터
a=np.array(range(1,101))
size=5          #time_steps=4(얼만큼 자를 것이냐)
#1,2,3,4,5//2,3,4,5,6//3,4,5,6,7//4,5,6,7,8//5,6,7,8,9//6,7,8,9,10
#데이터 자르기

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)] #행 구성
        aaa.append([item for item in subset]) #subset에 있는 아이템을 반환
    print(type(aaa))
    return np.array(aaa)
    #return 값이 np.array

dataset=split_x(a,size)
print("================")
print(dataset)

#외우기!!!
x=dataset[:96,0:4] # 행은 전체, 열은 0:4
y=dataset[:96,4] 
print(x)
print(y)

#reshape먼저 시켜주어야 함
#numpy로
x=x.reshape(96,4,1)
#여기서의 batch_size의 의미는 총 행(자를 것이 총 몇개인지)
#x=np.reshape(x,(6,4,1)) 똑같은 방법
print(x.shape)


# LSTM 모델을 완성하시오.

#2. 모델 불러오기 /전이학습
from keras.models import load_model

model=load_model(".//model//save_keras_44.h5") #load_model로 load된 애 불러오기

model.add(Dense(10,name='new1'))
model.add(Dense(10,name='new2'))
model.add(Dense(100,name='new3'))
model.add(Dense(1,name='new7'))


model.summary()

#name을 추가해주어야 충돌되지 않고 진행이 된다. 
#name을 쓰니 error가 잡히는 이유



# 행을 무시하는 이유는 훈련할때는 몇번했는지가 중요하지 않기 때문이다. 즉 데이터의 개수가 몇개인지는 중요하지 않다. 명시할 필요가 없다. 

#3. 훈련
model.compile(optimizer='adam',loss='mse',metrics=['acc']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss',patience=100, mode='auto')
#patience는 epoch보다 작아야 한다. 
hist=model.fit(x,y,epochs=1000,batch_size=1,callbacks=[early_stopping],validation_split=0.2) #model.fit을 변수에 할당해준다. 
print(hist) #<keras.callbacks.callbacks.History object at 0x000002579B0B2488> / 자료형만 보여줌--->그래프로 보기
print(hist.history.keys()) #dict_keys(['loss', 'mse'])

#실질적인 과정값 하나하나 보는 것은 중요하지 않다. 그래프로 그림을 그려서 전체적으로 보는 것이 좋다. 
#시각화를 잘하는 것이 good
import matplotlib.pyplot as plt 
plt.plot(hist.history['loss'],'r-') #x값은 자동으로 epoch가 될 것
plt.plot(hist.history['acc'],'b-') #metrix값
plt.plot(hist.history['val_loss'],'y-')
plt.plot(hist.history['val_acc'],'p-')
plt.legend(['train loss','train acc','val loss','val acc']) #범례(선에 대한 색깔과 설명)
plt.title("loss & acc")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show() #넣지 않으면 출력이 안된다. 

#val_loss>loss
#훈련시킨 것보다 val이 더 좋지 않다. 



#4. 평가, 예측
loss,acc=model.evaluate(x,y,batch_size=1) #metrics 꼭 써주어야 함
print("loss:",loss)
print("acc:",acc)
y_predict=model.predict(x)
print("y_predict:",y_predict)
