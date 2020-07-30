import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
# from keras.utils import to_categorical
#인코딩할 때는 sklearn OneHotEncoder를 이용한다.
#반대로 Decoder는 제공하지 않고 있어 디코딩할 때는 numpy의 argmax를 이용한다.

# y 분류하고 다시 y에 집어넣기
# 원핫인코딩-순서에 맞는 거에만 1, 나머지는 0--->분류모델에서 수행(반드시!!!)
# 원 핫 함수는 argmax함수와 함께 자주 사용한다.



# 2. 모델(다중분류)
x=np.array(range(1,11))
y=np.array([1,2,3,4,5,1,2,3,4,5]) 
y=y.reshape(-1,1) # 스칼라 10, 벡터 1
enc=OneHotEncoder()
enc.fit(y)
y_onehot=enc.transform(y).toarray()
#reshape(-1,정수)-->행의 위치에 -1이 들어있을 경우
# ex)x.shape=(3,4)
# x.reshape(-1,1) = x.shape(12,1)
# x.reshape(-1,2) = x.shape(6,2)
# x.reshape(-1,3) = x.shape(4,3)
print("y:",y_onehot)

# 슬라이싱
# encoder.fit(x)
# x_binned=encoder.transform(x)

print(x.shape)
print(y.shape) #10x6이 나온 이유??--->왜 됐는지와 제거(과제2)

model=Sequential()
model.add(Dense(10,input_shape=(1,),activation='relu')) #차원은 1개 #relu는 평타 85%이상의 성능
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(5,activation='softmax')) 


# 3. 훈련
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# 분류 모델에서는 accuracy
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실함수는 평균제곱오차 크로스 엔트로피 함수 사용
model.fit(x,y_onehot,epochs=1000,batch_size=5)

#4. 평가, 예측
loss,acc=model.evaluate(x,y_onehot,batch_size=1)  #loss, metrics에 집어넣은 값
print("loss:",loss)
print("acc:",acc)


x_pred=np.array([1,2,3])
y_predict=model.predict(x_pred)
print("y_predict.shape:",y_predict.shape)
print("y_predict:",y_predict)
#argmax는 softmax를 통해 나온 결과 중 최대값의 인덱스를 얻을 때 사용
print(np.argmax(y_predict,axis=1).reshape(-1,1)+1)
