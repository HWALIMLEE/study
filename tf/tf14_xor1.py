# 인공지능의 겨울...왕국
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# x,y,w,b,hypothesis,cost,train(optimizer)

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])
# print(x_data.shape) #(4,2)

w = tf.Variable(tf.random_normal([2,1]), name = 'weight') # 3만 일정, 뒤에는 변함
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b) # matmul - 행렬 연산을 해준다

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(cost)
# train = optimizer.minimize(cost)

# 준비만 하고 있는 것

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))
# tf.cast
# >> 조건에 따른 True, False의 판단 기준에 따라 True면 1, False면 0 반환


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(50001): #(epoch부분)
        cost_val, _ = sess.run([cost,optimizer],feed_dict={x:x_data,y:y_data})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    h, c,a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_data,y:y_data})
    print("\n Hypothesis:",h,"\n predicted:",c,
            "\n Accuracy:",a)

# hidden layer줌으로써 xor해결
