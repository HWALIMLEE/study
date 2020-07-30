# preprocessing

import tensorflow as tf
import numpy as np

def min_max_scaler(dataset): 
    numerator = dataset - np.min(dataset,0) # axis=0(열에서 최소값 찾겠다)
    denominator = np.max(dataset,0) - np.min(dataset,0)
    return numerator / (denominator + 1e-7) # 1e-7을 더한 이유 : 0으로 안만들기 위해서

dataset = np.array(
    
    [

        [828.659973, 833.450012, 908100, 828.349976, 831.659973],

        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],

        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],

        [816, 820.958984, 1008100, 815.48999, 819.23999],

        [819.359985, 823, 1188100, 818.469971, 818.97998],

        [819, 823, 1198100, 816, 820.450012],

        [811.700012, 815.25, 1098100, 809.780029, 813.669983],

        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],

    ]

)

dataset = min_max_scaler(dataset)
print(dataset)

x_data = dataset[:,0:-1] #(8,4)
y_data = dataset[:,[-1]] #(8,1)

print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,1])

# 회귀모델
w = tf.Variable(tf.random_normal([4,1]), name = 'weight') # 3만 일정, 뒤에는 변함
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b # matmul - 행렬 연산을 해준다

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = optimizer.minimize(cost)

# 준비만 하고 있는 것

mse = tf.reduce_mean(tf.losses.mean_squared_error(y,hypothesis))
# tf.cast
# >> 조건에 따른 True, False의 판단 기준에 따라 True면 1, False면 0 반환


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(50001):
        cost_val, _ = sess.run([cost,optimizer],feed_dict={x:x_data,y:y_data})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    h, a = sess.run([hypothesis,mse], feed_dict={x:x_data,y:y_data})
    print("\n Hypothesis:",h,
            "\n MSE:",a)
