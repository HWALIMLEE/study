#1. 데이터
import numpy as np
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([101,102,103,104,105,106,107,108,109,110])
y_test=np.array([101,102,103,104,105,106,107,108,109,110])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(50,input_dim=1,activation='relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1,activation='relu'))

#3. 요약
#parameter는 라인의 개수로 생각할 것
model.summary()

#4. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

#5. 평가, 예측
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_test,y_test))
loss,acc=model.evaluate(x_test,y_test,batch_size=1)
print("loss:",loss)
print("acc:",acc)

output=model.predict(x_test)
print("결과물:\n",output)
