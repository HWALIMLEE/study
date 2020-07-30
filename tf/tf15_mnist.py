import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()


print(x_train.shape) #(60000,28,28)
print(y_train.shape) #(60000,)
print(x_test.shape) #(10000,28,28)
print(y_test.shape)  #(10000,)

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

sess = tf.Session()
aaa = tf.one_hot(y_train,depth=10).eval(session=sess)
bbb = tf.one_hot(y_test,depth=10).eval(session=sess)

print(aaa.shape)
print(bbb.shape)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


# 첫번째 레이어
# 3 = hidden layer(내가 정하고 싶은대로)
w1 = tf.Variable(tf.random_normal([784,500]), name = 'weight1', dtype = tf.float32)
b1 = tf.Variable(tf.random_normal([500]), name = 'bias1',dtype = tf.float32)
layer1 = tf.matmul(x,w1) + b1

w2 = tf.Variable(tf.zeros([500,300]), name = 'weight2', dtype = tf.float32)
b2 = tf.Variable(tf.zeros([300]), name = 'bias2',dtype = tf.float32)
layer2 = tf.matmul(layer1,w2) + b2


w3 = tf.Variable(tf.zeros([300,200]), name = 'weight3', dtype = tf.float32)
b3 = tf.Variable(tf.zeros([200]), name = 'bias3',dtype = tf.float32)
layer3 = tf.matmul(layer2,w3) + b3


w4 = tf.Variable(tf.zeros([200,100]), name = 'weight4', dtype = tf.float32)
b4 = tf.Variable(tf.zeros([100]), name = 'bias4',dtype = tf.float32)
layer4 = tf.matmul(layer3,w4) + b4


w5 = tf.Variable(tf.zeros([100,80]), name = 'weight5', dtype = tf.float32)
b5 = tf.Variable(tf.zeros([80]), name = 'bias5',dtype = tf.float32)
layer5 = tf.matmul(layer4,w5) + b5

w6 = tf.Variable(tf.zeros([80,60]), name = 'weight6', dtype = tf.float32)
b6 = tf.Variable(tf.zeros([60]), name = 'bias6',dtype = tf.float32)
layer6 = tf.matmul(layer5,w6) + b6

w7 = tf.Variable(tf.zeros([60,50]), name = 'weight7', dtype = tf.float32)
b7 = tf.Variable(tf.zeros([50]), name = 'bias7',dtype = tf.float32)
layer7 = tf.matmul(layer6,w7) + b7


w8 = tf.Variable(tf.zeros([50,30]), name = 'weight8', dtype = tf.float32)
b8 = tf.Variable(tf.zeros([30]), name = 'bias8',dtype = tf.float32)
layer8 = tf.matmul(layer7,w8) + b8

w9 = tf.Variable(tf.random_normal([30,10]), name = 'weight9', dtype = tf.float32)
b9 = tf.Variable(tf.random_normal([10]), name = 'bias9',dtype = tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(layer8,w9) + b9)


# 마지막 최종 나가는 값 hypothesis

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-2).minimize(cost)

'''
tf.cast 0.5 보다 크면 true 0.5 보다 작거나 같으면 False

tf.cast
입력한 값의 결과를 지정한 자료형으로 변환해줌

tf.equal
tf.equal(x, y) : x, y를 비교하여 boolean 값을 반환
'''
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,tf.argmax(y,1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(101): #(epoch부분)
        _, hy_val, cost_val = sess.run([optimizer,hypothesis, cost],feed_dict={x:x_train,y:aaa})
        print(step,cost_val)

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:bbb}))
