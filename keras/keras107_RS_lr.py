# 100번을 카피해서 lr을 넣고 튠하시오.
# LSTM -> Dense로 바꿀 것

# 97번을 RandomizedSearchCV로 변경하시오
# score 빠짐-->채워넣기
from keras.datasets import mnist
from keras.utils import np_utils  #label이 시작하는게 0부터
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LSTM
import numpy as np
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Nadam

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("x_train.shape:",x_train.shape) #(60000,28,28)
print("x_test.shape:",x_test.shape) #(10000,28,28)

print("==================================")

# x_train=x_train.reshape(x_train.shape[0],28,28,1)/255 #0부터 255까지 들어있는 걸--->0부터 1까지로 바꿔줌(minmax)
# x_test=x_test.reshape(x_test.shape[0],28,28,1)/255

x_train=x_train.reshape(x_train.shape[0],28,28)/255
x_test=x_test.reshape(x_test.shape[0],28,28)/255

print(x_train.shape) #(60000,28,28)
print(x_test.shape)  #(10000,28,28)
print("=====================================")
y_train=np_utils.to_categorical(y_train) #y는 0부터 시작하기 때문에 np_utils써줘도 된다
y_test=np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000, 28*28)

print(y_train.shape)

print("====================================")

#2. 모델
# RandomSearch-->(model, hyperparameter, cv)
# RandomSearch의 parameter 의 model을 받기 위해서
# 모델 자체를 진짜 함수로 만든다 (앞으로 계속 이렇게 나올것)
# Dense모델 구성
# 모델을 여러번 쓸 수 있다
def build_model(drop=0.1, optimizer=Adam, learning_rate=0.1): # 초기값은 넣어주어야 함, 변수명 넣어줌
    inputs = Input(shape=(28*28,))
    x = Dense(512,activation='relu', name='hidden1')(inputs)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name='output')(x)
    
    opt = optimizer(lr=learning_rate) # optimizer 와 learning_rate 엮어주기
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['acc'])
    
    return model
#fit은 RandomSearch에서 함

def create_hyperparameters():
    batches = [256,128] #batch 종류
    optimizer = [Adam, SGD, RMSprop, Adadelta, Adagrad, Nadam] #optimizer의 종류
    learning_rate = [0.1,0.05,0.25,0.001]
    dropout = np.linspace(0.1, 0.5, 5) #0.1부터 0.5까지 5등분
    # epoch도 넣을 수 있고, node의 개수, activation도 넣을 수 있다(많이 넣을 수 있다.)
    # activation의 sigmoid와 softmax는 주의할 것
    return{"batch_size": batches, "optimizer" : optimizer, "drop" : dropout, "learning_rate" : learning_rate}

# 5*5*3=75번 돈다

# keras모델을 sickit learn 으로 싸는 거 wrapper
# 사이킷 런에서 쓸 수 있게 wrapping 한 거
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = build_model, verbose = 1)
model.fit(x_train,y_train,verbose=0)

hyperparameters=create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model,hyperparameters, cv=3) # 40000은 train/ 20000은 test/ 마지막 40000은?
#n_jobs=-1 넣으니까 터져버림(GPU까지 같이 쓰니까 터져버림)
search.fit(x_train,y_train,verbose=0)

# RandomSearch안에 keras넣기
# RandomSearch parameter에 model, parameter, cv 들어가는데 GridSearch는 sklearn이다
# keras모델을 넣어주기 위해 sklearn으로 감싸주는거

acc=search.score(x_test,y_test)

print("acc:",acc)
print(search.best_params_) #params_ = estimators_ (거의똑같다)
# best_params_와 best_estimators_의 차이 찾아보세용

# 시간 오래걸림
# 40000,20000,40000 의미 물어보기
# 랜덤서치가 가장 좋은 것 찾아온다( x )--->정말 random + 여기에서 best_params_써주어야 가장 좋은 거 나옴

# 경사하강법 lr(learning rate-학습률)
# >>>

#결과
"""
{'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 50}
"""

# 최적의 parameter값 찾아서 넣고 돌리기


# acc: 0.9620000123977661
# {'optimizer': <class 'keras.optimizers.Adam'>, 'learning_rate': 0.001, 'drop': 0.30000000000000004, 'batch_size': 256}