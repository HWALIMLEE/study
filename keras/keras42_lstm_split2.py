import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1. 데이터
a=np.array(range(1,101))
size=5          #time_steps=5(얼만큼 자를 것이냐)
#1,2,3,4,5//2,3,4,5,6//3,4,5,6,7//4,5,6,7,8//5,6,7,8,9//6,7,8,9,10
#데이터 자르기

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1): #size-1=x컬럼의 크기
        subset=seq[i:(i+size)] #행 구성
        aaa.append([item for item in subset]) #subset에 있는 아이템을 반환
    print(type(aaa))
    return np.array(aaa)
    #return 값이 np.array

dataset=split_x(a,size) #(96,5)
print("================")
print(dataset)

#외우기!!!
x=dataset[:90,0:4] # 행은 전체, 열은 0:4
y=dataset[:90,4] 

x_predict=dataset[90:96,0:4]
x=x.reshape(90,4,1)
x_predict=x_predict.reshape(6,4,1)

#reshape먼저 시켜주어야 함
#numpy로

#여기서의 batch_size의 의미는 총 행(자를 것이 총 몇개인지)
#x=np.reshape(x,(6,4,1)) 똑같은 방법
print(x_predict)
print(x.shape)


# 실습1. train,test분리
# 실습2. 마지막 6개의 행을 predict로 만들고 싶다. 
# train은 총 90개이다. 
# train과 test의 비율을 8:2
# validatio을 넣을 것(train의 20%)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=60,train_size=0.8)



# LSTM 모델을 완성하시오.
# 2. 모델
model=Sequential()

model.add(LSTM(10,input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(1)) #output은 무조건 dense
model.summary() 
# 행을 무시하는 이유는 훈련할때는 몇번했는지가 중요하지 않기 때문이다. 즉 데이터의 개수가 몇개인지는 중요하지 않다. 명시할 필요가 없다. 

# 3. 컴파일, 훈련
model.compile(optimizer='adam',loss='mse',metrics=['mse']) #metrics하나 안하나 상관없다.
from keras.callbacks import EarlyStopping 
early_stopping=EarlyStopping(monitor='loss',patience=70, mode='auto')
#patience는 epoch보다 작아야 한다. 
model.fit(x_train,y_train,epochs=1000,batch_size=1,callbacks=[early_stopping],validation_split=0.2,shuffle=True)
# 4. 평가, 예측
loss,mse=model.evaluate(x_test,y_test,batch_size=1) #metrics 꼭 써주어야 함
print("loss:",loss)
print("acc:",mse)

y_predict=model.predict(x_predict)
print("y_predict:",y_predict)

#Model, Sequential끼리 parameter 총 개수는 같다.