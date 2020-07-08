from sklearn.datasets import load_diabetes
import tensorflow as tf

sess = tf.Session()
datasets = load_diabetes()

x = tf.placeholder(tf.float32, shape=[None,10])
y = tf.placeholder(tf.float32, shape=[None,1])

x_data = datasets.data #(442,10)
y_data = datasets.target
y_data = y_data.reshape(442,1) #(442,1)

print(x_data.shape)
print(y_data)

# 회귀
w = tf.Variable(tf.random_normal([10,1]), name = 'weight') # 3만 일정, 뒤에는 변함
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b # matmul - 행렬 연산을 해준다

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.95)
train = optimizer.minimize(cost)

# 준비만 하고 있는 것

accuracy = tf.reduce_mean(tf.losses.mean_squared_error(y,hypothesis))
# tf.cast
# >> 조건에 따른 True, False의 판단 기준에 따라 True면 1, False면 0 반환


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(5001):
        cost_val, _ = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    h, a = sess.run([hypothesis,accuracy], feed_dict={x:x_data,y:y_data})
    print("\n Hypothesis:",h,
            "\n Accuracy:",a)
