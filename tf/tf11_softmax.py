import tensorflow as tf
import numpy as np

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

W = tf.Variable(tf.random_normal([4, 3]),name='weight') # 입력받는  x의 shape 4, 출력되는 y의 shape 3
b = tf.Variable(tf.random_normal([3]),name='bias') # 1 or 3 or (1,3) ===> answer : 3 (1을 넣었을 때도 나오기는 하지만 cost값이 높게 나온다)

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b) # softmax는 값 합쳐서 1이 나와야함 (keras110_9)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # categorical crossentropy 풀어쓴 것

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, hy_val,cost_val = sess.run([optimizer,hypothesis, cost],feed_dict={x:x_data,y:y_data})

        if step % 200==0:
            print(step, cost_val)


# 최적의 W와 b가 구해진다.
# Predict
    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]}) 
    print(a,sess.run(tf.argmax(a,1))) # 1은 행을 의미 [0]

    b = sess.run(hypothesis, feed_dict={x:[[1,3,4,3]]})
    print(b,sess.run(tf.argmax(b,1))) # 1은 행을 의미 [1]

    c = sess.run(hypothesis, feed_dict={x:[[11,33,4,13]]})
    print(c,sess.run(tf.argmax(c,1))) # 1은 행을 의미 [1]

    print(c.shape)
    
    all = sess.run(hypothesis,feed_dict={x:[[1,11,7,9],[1,3,4,3],[11,33,4,13]]})
    print(all,sess.run(tf.argmax(all,1)))


# import numpy as np
# a = np.array([[1,2,3],[4,5,6]]) #(2,3)
# b = np.array([10,20,30])        #(3,)
# c = np.array([[10,20,30]])      #(1,3)   
# print(a.shape)
# print(b.shape)
# print(c.shape)