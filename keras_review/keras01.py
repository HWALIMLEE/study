#데이터 준비
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(3))
model.add(Dense(1,activation='relu'))

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit(x,y,epochs=100,batch_size=1)

loss,mse=model.evaluate(x,y,batch_size=1)

print("loss:",loss)
print("mse:",mse)
