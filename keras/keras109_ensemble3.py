
# 동일한 데이터 , 한개의 모델
# 1. 데이터
import numpy as np
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(1,))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)


merge = concatenate([x1,x2])  # 모델 2개 합침

x3 = Dense(100)(merge)
output1 = Dense(1)(x3)

x4 = Dense(100)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs=[input1, input2], outputs=[output1, output2]) 

model.summary()


#3. 컴파일, 훈련
model.compile(loss = ['mse','binary_crossentropy'], optimizer = 'adam',
                        loss_weights=[0.1,0.9], # 두 개 이상이니까 가능/ 뒤쪽 분류 모델이 너무 안맞아서 이쪽에 가중치를 더 주려고
                        metrics=['mse','acc'])
# binary_crossentropy===> 이진분류 일때 loss 지표값
# 분기 시킨다
# loss값 7개 나온다.
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate([x1_train,x2_train],[y1_train,y2_train])
print("loss:",loss)
x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])
# 11,12,13,14의 회귀값과 분류값 동시에 나옴
# 이 값이 어떻게 나오는 지 분석

y_pred = model.predict([x1_pred, x2_pred])
print("y_pred:",y_pred)

# 전체 dense의 loss = dense5의 loss + dense8의 loss
# 지표 1개로 모델 1개 돌리는 게 제일 Best


"""
            전체 loss           dense8_loss           dense11_loss           dense8_mse   dense8_acc   dense11_mse       dense11_acc
 loss: [0.6179057359695435, 0.002910121576860547, 0.6862386465072632, 0.002910121576860547, 1.0, 0.24662837386131287, 0.6000000238418579]
 """


# loss_weight를 주어도 딱히 좋아지지 않는다.
# activation의 default 값은 linear

