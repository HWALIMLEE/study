"""
#2번의 첫번째 답
# x=[1,2,3] #list
# x=x-1
# print(x) #numpy안쓰고 list형태로 쓰면 TypeError뜬다.
"""

#인공지능에서는 numpy 활용도 굳
#그러나 numpy는 한가지 자료형만 쓸 수 있다.
import numpy as np
y=np.array([1,2,3,4,5,1,2,3,4,5])

"""
y=y-1 #넘파이에서만 가능

print(y) #[0 1 2 3 4 0 1 2 3 4] #인덱스를 그냥 0으로 맞춰버림

from keras.utils import np_utils
y=np_utils.to_categorical(y)
print(y)
print(y.shape)
#어차피 argmax는 인덱스값 반환하는 것이기 때문
"""

#2번의 두번째 답
from sklearn.preprocessing import OneHotEncoder
aaa=OneHotEncoder()
y=y.reshape(-1,1)
aaa.fit(y)
y=aaa.transform(y).toarray()
print(y)
print(y.shape)
#fit.transform--->scaler에서 나왔음
#OneHotEncoder의 단점
#OneHotEncoder는 y의 차원을 바꿔주어야 한다. 
#reshape(-1,1) = reshape(10,1)  -1은 인덱스의 끝--->2차원으로 바꿔준다.

#OneHotEncoder은 sickitlearn은 차원 맞춰주어야 하고 , keras(to_categorical)는 인덱스 0부터 시작하는 거 맞춰주기
#다중분류에서는 OneHotEncoder 필수로 사용

