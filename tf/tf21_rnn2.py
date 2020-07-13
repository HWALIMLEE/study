import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape) #(10,)

# RNN 모델을 짜시오!
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)


dataset_new = split_x(dataset, 6)
x_data = dataset_new[:,:5]
y_data = dataset_new[:,5]

print(x_data.shape) #(5,5)
print(y_data.shape) #(5,)
#(1,5,5) 보다는 (5,5,1)이 낫다

x_data = x_data.reshape(1,5,5)
y_data = y_data.reshape(1,5)
print(x_data)
print(y_data)



sequence_length= 5
input_dim = 5
output = 20 # 다음 노드로 전달하는 노드 개수
batch_size = 1 # 전체 행


X = tf.placeholder(tf.float32, shape=[None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, shape=[None, sequence_length])


cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) 
# 위 두 줄이 한줄의 keras 와 같다.
print(hypothesis) 

# 3-1. 컴파일 
# optimizer 
# ones는 선형?
weights = tf.ones([batch_size,sequence_length]) # y의 shape를 weight로 넣어주기 #(1,5)
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

prediction = hypothesis

# 3-2. 훈련(fit부분)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost,train],feed_dict={X : x_data,Y : y_data})
        result = sess.run(prediction, feed_dict={X:x_data}) # 이번에는 train, test 나누지 않음
        print(i,"loss:",loss,"prediction:", result ,"true Y:",y_data)

# tensorflow는 거꾸로 올라간다
          
