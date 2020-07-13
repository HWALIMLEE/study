import tensorflow as tf
import numpy as np

# if문 알고리즘으로는 힘들다
# hi hell : x
# i hello : y

# 1. 데이터
# data = hihello
# 인덱스 넣어주기 위해 한글자씩 떼어주기
idx2char=['e','h','i','l','o']

_data = np.array([['h','i','h','e','l','l','o']],dtype=np.str).reshape(-1,1)
print(_data.shape) # (1,7)  
print(_data)
print(type(_data)) # <class 'numpy.ndarray'

# one-hot-encoding

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print("========================================================")
print(_data.dtype) # float64
print(_data)       
"""
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
 """
print(type(_data)) # <class 'numpy.ndarray'>

"""
OneHotEncoder로 바꿔주기 위해서는 
shape가 (1,7)이 아니고
shape가 (7,1)이 어야 한다. 
"""

# x와 y나누기
x_data = _data[:6,] # x는 h부터 l까지
y_data = _data[1:,] # y는 i부터 o까지

print("============x============")
print(x_data)
print("============y==========")
print(y_data)
print("========================")

# LSTM
# (행, 열, 몇개씩 자를지)
# 현재 shape=(1,6,5)
# input_shape=(6,5)
# tensorflow의 shape=(1,6)


print('===========y argmax===================')
y_data = np.argmax(y_data,axis=1)
print(y_data)  #[2 1 0 3 3 4]
print(y_data.shape) #(6,)
y_data = y_data.reshape(1,6)
print(y_data.shape) #(1,6)
print(y_data) #[[2 1 0 3 3 4]]

x_data = x_data.reshape(1,6,5)
print(x_data.shape)
print(x_data)
"""
[[[0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 0.]
  [0. 1. 0. 0. 0.]
  [1. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0.]
  [0. 0. 0. 1. 0.]]]
  """

sequence_length= 6
input_dim = 5
output = 5 # 다음 노드로 전달하는 노드 개수
batch_size = 1 # 전체 행

# X = tf.placeholder(tf.float32, shape=[None,sequence_length,input_dim]) # <=> X=tff.placeholder(tf.float32, (None,sequence_length,input_dim))
# Y = tf.placeholder(tf.float32, shape=[None,sequence_length])

X = tf.compat.v1.placeholder(tf.float32, shape=[None,sequence_length,input_dim]) # <=> X=tff.placeholder(tf.float32, (None,sequence_length,input_dim))
Y = tf.compat.v1.placeholder(tf.int32, shape=[None,sequence_length])

print("==================placeholder print========================")
print(X)  # Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(Y)  # Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)


#2. 모델구성
# keras : model.add(LSTM(100, input_shape=(6,5)))
# tensorflow : cell = tf.nn.rnn_cell.BasicLSTMCell(output)


# cell = tf.nn.rnn_cell.BasicLSTMCell(output) # <=> model.add(LSTM(100))
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # <=> model.add(LSTM(100,input_shape=(6,5)))
# 위 두 줄이 한줄의 keras 와 같다.
print(hypothesis) # Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32) # 전 output이 몇개씩 자르는 지로 들어감

# 3-1. 컴파일 

# optimizer 
# ones는 선형?
weights = tf.ones([batch_size,sequence_length]) # y의 shape를 weight로 넣어주기
# LSTM쓰게 되면 loss를 이 형식으로 써야 한다. 
sequence_loss = tf.contrib.seq2seq.sequence_loss(
 logits = hypothesis, targets = Y, weights = weights
)
# hypothesis와 Y값을 비교하는 loss

# 예측값과 원값이 동일하면 acc:1, loss:0
# loss가 최저가 되는 지점은 예측값-원값=0
# accuracy=1
cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# Value passed to parameter 'labels' has DataType float32 not in list of allowed values: int32, int64 
# label = Y
# target값이 int형만 먹힌다?

# 문자 변환 역으로 하면 된다. 
prediction = tf.argmax(hypothesis, axis=2) # (1,6,5) 1은 axis=0, 6은 axis=1, 5는 axis=2 # 삼차원 axis
print("predictin:",prediction)

# 3-2. 훈련(fit부분)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
        result= sess.run(prediction, feed_dict={X:x_data}) # 이번에는 train, test 나누지 않음
        print(i,"loss:",loss,"prediction:", result ,"true Y:",y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPredction Str:","".join(result_str)) # 문자로 출력하겠다
# tensorflow는 거꾸로 올라간다
          