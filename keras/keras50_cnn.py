# 숫자로 바꿔서 계속 비슷한 것 찾기--기계가
# 이미지를 자르는 것이 제일 기초

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense,Flatten

model=Sequential() #convolusion해서 나가는 값이 10개
model.add(Conv2D(10,(2,2),input_shape=(10,10,1))) #(9,9,10) # input_shape(,,1)-가로, 세로,흑백/input_shape(,,3)-가로, 세로,컬러
# 가로, 세로 2만큼씩 자르겠다.
# 10개가 밖으로 나간다. 
# CNN은 4차원(10000장, 10,10,1) --->input_shape=(10,10,1) 이미지 장수와 상관없이
# input_shape는 그림이 들어간다. 가로x세로x명암
# 자른 걸 또 자른다. 

model.add(Conv2D(7,(3,3))) #(7,7,7) #input_shape명시 안해줘도 됨/ 또 특성을 찾기
model.add(Conv2D(5,(2,2),padding="same")) #(7,7,5)
model.add(Conv2D(5,(2,2))) #(6,6,5)
# model.add(Conv2D(5,(2,2),strides=2)) #(3,3,5)
# model.add(Conv2D(5,(2,2),strides=2,padding="same")) #(3,3,5) #stride가 우선순위

model.add(MaxPooling2D(pool_size=3)) #이전 shape/pool_size--->ex)(6,6,5)/3=(2,2,5)
#padding='same'을 지우니 summary값이 달라진다. 

model.add(Flatten())
model.add(Dense(1))

model.summary()
# 차원은 4차원
# 점심시간에 파악하기
# CNN방식은 이미지를 조각으로 쪼갠다.
# 현재는 짜르고 증폭까지 되어있음
# 이미지를 레이어마다 짜를 수 있다. --->input_shape는 4차원, 3차원만 명시

"""
conv2D(filter,kernel_size((,)),input_shape=(height,width,channel))
kernal_size=(2,2) = kernel_size=2--->둘 다 가로세로 2로 자르겠다는 의미
(batch_size, height, width, channel)
"""
#데이터 잘라서 붙이면 증폭된다. 
