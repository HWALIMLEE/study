# multi variable
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,2]), name = 'weight') # 3만 일정, 뒤에는 변함
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b) # matmul - 행렬 연산을 해준다

# sigmoid통과하게 됨

"""
(5 * 3) * (3 * 1)==[5 * 1]
"""

# cost = tf.reduce_mean(tf.square(hypothesis - y))

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*   #crossentropy loss function
                            tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.99)
train = optimizer.minimize(cost)

# 준비만 하고 있는 것
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))
# tf.cast
# >> 조건에 따른 True, False의 판단 기준에 따라 True면 1, False면 0 반환


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언

    for step in range(5001):
        cost_val, _ = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data,y:y_data})
    print("\n Hypothesis:",h,"\n Correct(y):",
            "\n Accuracy:",a)



# with문 안썼으니까 sess.close()해주어야 한다
# 케라스보다 연산이 조금 더 빠르다
