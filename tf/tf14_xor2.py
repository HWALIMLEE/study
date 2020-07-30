import tensorflow as tf
import numpy as np


tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)



# x,y,w,b hypothesis, cost, train
# sigmoid

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 첫번째 레이어
# 3 = hidden layer(내가 정하고 싶은대로)
w1 = tf.Variable(tf.random_normal([2,100]), name = 'weight1', dtype = tf.float32)
b1 = tf.Variable(tf.random_normal([100]), name = 'bias1',dtype = tf.float32)
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)

# 위의 연산을 케라스로 한다면 ==> model.add(Dense(3, input_dim=2))

# 두번째 레이어
w2 = tf.Variable(tf.zeros([100,50]), name = 'weight2', dtype = tf.float32)
b2 = tf.Variable(tf.zeros([50]), name = 'bias2',dtype = tf.float32)
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.random_normal([50,1]), name = 'weight3', dtype = tf.float32)
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3',dtype = tf.float32)
hypothesis = tf.sigmoid(tf.matmul(layer2,w3) + b3)


# 마지막 최종 나가는 값 hypothesis

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y)* tf.log(1-hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-2)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis >=0.5, dtype=tf.float32)
'''
tf.cast 0.5 보다 크면 true 0.5 보다 작거나 같으면 False

tf.cast
입력한 값의 결과를 지정한 자료형으로 변환해줌

tf.equal
tf.equal(x, y) : x, y를 비교하여 boolean 값을 반환
'''
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess: # Session을 close하지 않으려고
    sess.run(tf.global_variables_initializer())

    for step in range(50001):
        cost_val, _ , acc= sess.run([cost, train,accuracy], feed_dict={x: x_data, y: y_data})

        if step % 200 ==0:
            print(step, cost_val, acc)
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})

    print('\n Hypothesis : ', h ,'\n Correct (y) : ', c , '\n Accuracy : ', a)