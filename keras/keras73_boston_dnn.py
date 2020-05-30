from sklearn.datasets import load_boston
boston=load_boston()

#1. 데이터
"""
data : x값
target : y값
"""
dataset=load_boston()
x=dataset.target()
y=dataset.target()

print("x_train.shape:",x.shape)
print("x_test.shape:",x.shape)
print("y_train.shape:",x.shape)
print("y_test.shape:",x.shape)

#구조 알아보기, 분리, 스켈링