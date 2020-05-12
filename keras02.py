#모듈 불러오기
from keras.models import Sequential #케라스에 있는 순차적인 모델 구현
from keras.layers import Dense
import numpy as np

#데이터 준비
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([101,102,103,104,105,106,107,108,109,110])
y_test=np.array([101,102,103,104,105,106,107,108,109,110])

#모델 구성
model=Sequential()
model.add(Dense(5,input_dim=1, activation='relu')) #dimension차원, 데이터 한개만 넣었기 때문에 일차원
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1,activation='relu'))

#모델 요약
model.summary()
#39번 연산했다
#총 파라미터 값 
#파라미터는 라인의 개수로 생각할 것


#compile과 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_test, y_test))

#평가,예측
loss,acc=model.evaluate(x_test, y_test, batch_size=1)

print("loss:",loss)
print("acc:",acc)

output=model.predict(x_test)
print(output)


