import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)


x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255

# session열면 닫아야 함
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])


learning_rate = 0.01
training_epochs = 15
batch_size = 100
total_batch = x_train.shape[0]//batch_size # 60000/100 #600

# x = tf.placeholder(tf.float32, shape=[None,784])
# y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) #dropout


w1 = tf.get_variable("w1",shape=[784,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(tf.matmul(x,w1) +b1)
L1 = tf.nn.dropout(L1,keep_prob = keep_prob)

w2 = tf.get_variable("w2",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2,keep_prob = keep_prob)

w3 = tf.get_variable("w3",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b2)
L3 = tf.nn.dropout(L3,keep_prob = keep_prob)

w4 = tf.get_variable("w4",shape=[512,256],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4,keep_prob = keep_prob)

w5 = tf.get_variable("w5",shape=[256,10],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4,w5)+b5)


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # categorical crossentropy 풀어쓴 것

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs): # 15
    avg_cost = 0 
    for i in range(total_batch):    # 600
        batch_xs, batch_ys = x_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size] # wrong
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob : 0.7} # 0.7만큼 남기겠다
        c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += c/total_batch
    print("Epoch:",'%04d' % (epoch + 1),'cost=','{:.9f}'.format(avg_cost))
# print("훈련 끗")

prediction = tf.equal(tf.arg_max(hypothesis,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test, keep_prob:0.7})) ##acc출력할 것