from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM


#1. 데이터
x=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #4행 3열
y=array([4,5,6,7]) #(4,)--->스칼라가 4개짜리 벡터1개 (4,)!=(4,1) 절대 (4,1)이라고 하면 안된다. input_dim=1(일차원)
# y2=array([[4,5,6,7]]) #(1,4)
# y3=array([[4],[5],[6],[7]]) #(4,1)

print("x.shape:",x.shape)  #(4,3)
print("y.shape:",y.shape)  #(4,)  --->(4,1)이라고 하면 에러 난다. 
#shape해서 확인해보기!!
#자르는 숫자 명시 ex)4x3x1--> (4,3)을 1개씩 연속된 데이터 계산하겠다(1개씩 작업) (행, 열, 몇개로 자를건지)

# x=x.reshape(4,3,1) #전체 데이터는 변경되지 않는다. 
# reshape할 때 검사는 곱하기! (4*3)=(4*3*1)
x=x.reshape(x.shape[0],x.shape[1],1) # x.shape[0]=4, x.shape[1]=3 
#위에 식이나 아래식이나 결과는 동일하나 정석은 두번째꺼가 맞는 것!

print("x:",x.shape)
print("x:",x)

#2. 모델구성
# LSTM은 DENSE모델에 비해 많은 연산을 하게 된다. 
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1))) #시계열 input_shape=(3,1) ***행 무시***, LSTM에서 중요한 것: 컬럼의 개수와 몇개씩 잘라서 계산할 것이냐, 행은 중요하지 않다
#여기서부터는 Dense모델
model.add(Dense(5))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(50))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(1)) #하나 예측 y=[4,5,6,7]

model.summary() #param[1]=480
"""
#과제1
#param 이 왜 480나오는 지 찾아보기
#input_shape는 (3,1)밖에 안들어갔는데 왜 480이 나올까
"""
#3. 실행
model.compile(optimizer='adam',loss='mse') #metrics하나 안하나 상관없다.
model.fit(x,y,epochs=300,batch_size=1)

x_input=array([5,6,7]) #(3,) 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)
x_input=x_input.reshape(1,3,1)
print(x_input)

yhat=model.predict(x_input)
print(yhat)
##정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정



#예제
# x=array([[1,2,3],[1,2,3]]) #(2,3)
# print(x.shape)
# y=array([[[1,2],[4,2]],[[4,5],[5,6]]]) #(덩어리 개수, 개수, 제일 작은 단위) #작은거부터 치고 올라가기
# print(y.shape)
# z=array([[[1],[2],[3]],[[4],[5],[6]]])
# print(z.shape)

# w=array([[[1,2,3,4]]])
# print(w.shape)
# k=array([[[[1],[2]]],[[[3],[4]]]])
# print(k.shape)
###스칼라    벡터    행렬     텐서

