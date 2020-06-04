from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.layers import Dense
from keras. models import Sequential
import numpy as np

#머신러닝에서는 그냥 하면 되는데
#딥러닝은 np.array로 변경
#딥러닝은 가중치의 곱의 합
#행렬 곱, 행렬 연산 잘하기 위해
#list는 appending될 뿐-->연산 자체가 안 이루어진다.
#머신러닝은 가중치 연산이 아니다. 따라서 리스트도 가능하다
#labelencoder

#1.데이터
x_data=[[0,0],[1,0],[0,1],[1,1]]
y_data=[0,1,1,0]
x_data=np.array(x_data)
print(x_data)
y_data=np.array(y_data)

print("x_data.shape",x_data.shape) #(4,2)
print("y_data.shape:",y_data.shape) #(4,)


#2.모델
# model=LinearSVC()
# model=SVC()
# lin = LinearSVC()
# sv = SVC()
# kn = KNeighborsClassifier(n_neighbors=1)
model=Sequential()
#n_neighbors 작을수록 더 치밀
#데이터가 적을수록 n_neighbors 적게 하는 것이 좋다
#각 개체를 한개씩만 연결하겠다


model.add(Dense(10,input_dim=2,activation='relu')) #input과 아웃풋 #딥러닝 아님
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid')) #마지막에만 시그모이드
#output dimension=1



#3.실행
model.compile(optimizer='adam',metrics=['acc'],loss='binary_crossentropy') #metrics는 결과만 보는 것
model.fit(x_data,y_data,epochs=100,batch_size=1)
loss,acc=model.evaluate(x_data,y_data)
#accuracy는 1이 나올 수 없다, 선형으로는 절대 나올 수 없쥐


#4.평가예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test=np.array(x_test)


y_predict = model.predict(x_test)

# acc=accuracy_score([0,1,1,0],y_predict)
#그냥 score는 evaluate와 동일한 것
#evaluate대신 score사용

# acc2=accuracy_score([0,1,1,0],y_predict)


print(x_test,"의 예측 결과:",y_predict)

print("acc=",acc) 