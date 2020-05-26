# 1. 데이터
import numpy as np
x=np.array(range(1,11))
y=np.array([1,0,1,0,1,0,1,0,1,0]) #--->홀수면 1, 짝수면 0
# 결과값이 두가지로 한정(이진분류)
# 0.5를 기준으로 1과 0으로 분류

# 2. 모델
from keras.layers import Dense
from keras.models import Sequential

#activation은 다 들어갈 수 있다. 
model=Sequential()
model.add(Dense(100,input_dim=1,activation='relu')) #차원은 1개 #relu는 평타 85%이상의 성능
model.add(Dense(200,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid')) # 마지막 나온 최종 값 x 시그모이드--->0 or 1
#activation default값: 

"""
시그모이드 함수란
-->시그모이드 함수는 단계 함수와 비슷하지만 조금 더 부드러운 변화로 구분을 하게 된다. 분류 모델. 
여기서 사용하는 최적화 알고리즘은 기울기 상승을 이용하여 찾게 된다.(Gradient Ascent)-->이 알고리즘은 함수에서 최대 지점을 찾기 위해 기울기의 방향으로 이동하는 거시 가장 좋은 방법에 기반
***참고로 기울기 하강은 상승과는 다르게 최대값을 찾고자 하는 것이 아닌 최소값을 찾고자 하는 것
"""

# 3. 훈련
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) #loss에 들어가는 방식이 mse-->binary_crossentropy #10개 중에 1과 0을 얼마나 잘 맞추느냐-->accuracy
#분류 모델에서는 accuracy
#이진 분류에서는 loss값이 binary_crossentropy밖에 없음
model.fit(x,y,epochs=1000,batch_size=1)

#4. 평가, 예측
loss,acc=model.evaluate(x,y,batch_size=1)  #loss, metrics에 집어넣은 값
print("loss:",loss)
print("acc:",acc)


# 과제1.
# y_predict값이 0과 1로 나오게끔 하기--->regression사용하기
# def sigmoid(z):
#     if z>0.5:
#         z=1
#     else:
#         z=0
#     return z

x_pred=np.array([1,2,3])
y_pred=model.predict(x_pred)
# for i in range(len(y_pred)):
#     a=sigmoid(y_pred[i])
#     answer=[]
#     answer.append(a)
#     print("answer:",answer)

y_pred=np.where(y_pred>=0.5,1,0)
print("y_pred",y_pred)