# callback의 기능은 더 많다. + tensorboard
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
#데이터 삭제
#모델을 저장

#2.모델
model=Sequential()

model.add(LSTM(10,input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(10)) #output 을 바꿈 keras45파일에서 오류난다. 
model.summary() 


# model만 summary되서 나온다. 
# 저장(model만)
# 쓸만한 모델만 저장할 수 있다. 

model.save(".//model//save_keras_44.h5") 
# model.save("./model/save_keras_44.h5")
# model.save(".\model\save_keras_44.h5")
# 세개 모두 저장 잘 된다. 


#.은 현재 directory
#모델 확장명은 h5  --->어디에 저장이 될까?(경로 지정 안해주면 기본 study폴더에 저장이 된다.)

print("저장 잘됐다.")#-->이 소스가 문제가 없다면 출력될 것
