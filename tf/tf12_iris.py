# 다중분류
# iris코드를 완성하시오.
from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.model_selection import train_test_split

# test값으로 hypothesis

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(150,1)
print(x_data.shape) #(150,4)
print(y_data.shape) #(150,1)

sess = tf.Session()
# # 원핫인코딩
aaa = tf.one_hot(y_data,depth=3).eval(session=sess)
print(aaa.shape) #(150,1,3)
aaa = aaa.reshape(150,3)


x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

W = tf.Variable(tf.random_normal([4,3]),name='weight') 
b = tf.Variable(tf.random_normal([3]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,W)+b) # matmul - 행렬 연산을 해준다

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
# train = optimizer.minimize(cost)

x_train, x_test,y_train,y_test = train_test_split(x_data, aaa,test_size=0.2, random_state=55)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 초기화==선언
    for step in range(50001): #(epoch부분)
        _, hy_val, cost_val = sess.run([optimizer,hypothesis, cost],feed_dict={x:x_train,y:y_train})
        if step % 100==0:
            print(step, cost_val)
    # 실제로 실현되는 부분
    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
    
    #정확도
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))


    
    # GYU code
    # predicted = tf.arg_max(hypo,1)
    # acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))
    
