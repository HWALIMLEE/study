from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1.데이터
x_data=[[0,0],[1,0],[0,1],[1,1]]
y_data=[0,1,1,0]
#and

#2.모델
model=LinearSVC()
#레이어 늘림
#두번 접는 거


#3.실행
model.fit(x_data,y_data)

#4.평가예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

acc=accuracy_score([0,1,1,0],y_predict)
#그냥 score는 evaluate와 동일한 것
#evaluate대신 score사용

print(x_test,"의 예측 결과:",y_predict)
print("acc=",acc)



#xor 문제는, 데이터를 분류해낼 '선'이 부족한 것입니다.

#그렇다면 선을 늘리면 됩니다.