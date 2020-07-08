import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [3.,5.,7.]

w = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
cost_history = []

with tf.Session() as sess:
    for i in range(-30,50):
        curr_w = i*0.1 # 그림 그릴 간격들
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

plt.plot(w_history, cost_history)
plt.show()


# 세로축은 cost
# 가로축은 weight

