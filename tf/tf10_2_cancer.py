from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,1])

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(569,1)


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)

print(x_data.shape) #(569, 30)
print(y_data.shape) #(569, 1)

# 회귀
w = tf.Variable(tf.random_normal([30,1]), name = 'weight') # 3만 일정, 뒤에는 변함
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b) # matmul - 행렬 연산을 해준다

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000005)
train = optimizer.minimize(cost)

# 준비만 하고 있는 것
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))
# tf.cast
# >> 조건에 따른 True, False의 판단 기준에 따라 True면 1, False면 0 반환


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(5001):
        cost_val, _ = sess.run([cost,train],feed_dict={x:x_train,y:y_train})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    h, c,a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test,y:y_test})
    print("\n Hypothesis:",h,"\n predicted:",c,
            "\n Accuracy:",a)
