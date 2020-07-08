# tf06_1.py를 카피해서
# lr을 수정해서 연습
# 0.01 -> 0.1/0.001/1
# epoch가 2000번보다 적게 줄여보자

import tensorflow as tf
tf.set_random_seed(777) # randomseed로 고정

# sess = tf.Session()
x_train = tf.placeholder(tf.float32, shape=[None])  # placeholder는 sess.run할 때 집어넣는다(input과 비슷한 개념) ===> sess.run에 feed_dict가 들어간다/ placehoder는 출력되는 값 자체는 없다
y_train = tf.placeholder(tf.float32, shape=[None])


# 변수// 0~1 사이의 정규확률분포 값을 생성해주는 함수
W = tf.Variable(tf.random_normal([1]),name='weight') #2로 바꾸니 에러가 난다. Dimensions must be equal, but are 3 and 2 for 'mul' (op: 'Mul') with input shapes: [3], [2]
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x_train * W + b

sess = tf.Session()

cost = tf.reduce_mean(tf.square(hypothesis-y_train)) # MSE
 
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # optimizer// 최소의 loss가 최적의 weight값 구해줌

with tf.Session() as sess: #tf.Session 보다는 tf.compat.v1.Session(version문제)
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],feed_dict={x_train:[1,2,3],y_train:[3,5,7]}) # train은 하되 결과값은 출력하지 않겠다.// cost를 minimize

        if step % 20 ==0:
            print(step, cost_val, W_val, b_val)

    print("예측:",sess.run(hypothesis, feed_dict={x_train:[4]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))


