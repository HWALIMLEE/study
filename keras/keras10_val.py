#과제: R2를 음수가 아닌 0.5이하로 줄이기
#레이어는 인풋과 아웃풋을 포함 5개 이상(히든이 3개 이상), 노드는 레이어당 각각 5개 이상
#batch_size=1
#epochs=100이상

#1. 데이터
import numpy as np
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([11,12,13,14,15])
y_test=np.array([11,12,13,14,15]) #테스트 데이터는 모델 성능에 영향 미치지 않는다. 
# x_pred=np.array([16,17,18])
x_val=np.array([101,102,103,104,105])
y_val=np.array([101,102,103,104,105]) #weight 값 유추할 수 없다. 불연속적 데이터
# train 데이터는 1~10, 101~105


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(50,input_dim=1)) #x,y한덩어리(input_dim=1)
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(150))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) 여기서부터 실행이 안 됨 
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(1))

#3.훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse']) 
model.fit(x_train, y_train, epochs=50, batch_size=1,
validation_data=(x_val, y_val)) #validation값 fit에 적용
# metrics에 뜨는 것은 loss, mse, val_loss, val_mse
# val_loss가 loss보다 통상적으로 더 낮다. 
# batch_size가 낮다고 해서 꼭 좋은 loss값이 나오는 것은 아니다. #계속 시행했을 때 acc=1.0나오면 좋은 값 #훈련train


# 4.평가
loss,mse=model.evaluate(x_test,y_test,batch_size=1) #model.evaluate 기본적으로 compile에서 설정한 loss, metrics반환하는 함수 #evaluate는 test값 평가 #그런데 같은 데이터 값으로 다시 평가했음 #과적합
print("loss:",loss)
print("mse:",mse)

#5.예측
#y_pred=model.predict(x_pred)
#print("y_predict:",y_pred)
#훈련데이터와 평가용 데이터는 같은 데이터 쓰게 되면 안된다. 과적합

y_predict=model.predict(x_test)
print("y_predict:",y_predict)


#RMSE구하기
from sklearn.metrics import mean_squared_error as mse

#함수는 재사용
#y_test가 원래 값 mse=시그마(y-yhat)^2/n
#y_predict는 x_test로 예측

def RMSE(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))
#RMSE는 함수명
print("RMSE:",RMSE(y_test,y_predict))
#RMSE는 가장 많이 쓰는 지표 중 하나

#R2구하기
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

#즉, RMSE는 낮게 R2는 높게
