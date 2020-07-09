import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)


x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255

# session열면 닫아야 함
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])

w = tf.Variable(tf.zeros([28*28,10]),name="weight")
b = tf.Variable(tf.zeros([10]),name="bias")
layer = tf.nn.softmax(tf.matmul(x,w)+b)
# layer = tf.nn.elu(tf.matmul(x,w)+b)
# layer = tf.nn.selu(tf.matmul(x,w)+b)
# layer = tf.nn.relu(tf.matmul(x,w)+b)

#model.add(Dense(100,input_shape=(2,)))

# w = tf.Variable(tf.zeros([50,10]),name="weight")
# b = tf.Variable(tf.zeros([10]),name="bias")
# layer = tf.nn.softmax(tf.matmul(layer,w)+b)
#model.add(Dense(50))

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(layer),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(loss)
# train = optimizer.minimize(cost)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for step in range(100):
        _,loss_val=sess.run([optimizer,loss],feed_dict={x:x_train,y:y_train})
        if step % 5==1:
        #     print(loss_val)
            print(f"step:{step},loss_val:{loss_val}")
        # 실제로 실현되는 부분
    correct_prediction = tf.equal(tf.argmax(layer,1),tf.argmax(y,1))
    
    #정확도
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))

""" 
step:1,loss_val:1.5618211030960083
step:6,loss_val:1.1636066436767578
step:11,loss_val:0.7512286901473999
step:16,loss_val:0.6932064294815063
step:21,loss_val:0.5682181119918823
step:26,loss_val:0.4837920665740967
step:31,loss_val:0.4415675699710846
step:36,loss_val:0.42330166697502136
step:41,loss_val:0.4117785692214966
step:46,loss_val:0.4026491343975067
step:51,loss_val:0.39488503336906433
step:56,loss_val:0.3881279230117798
step:61,loss_val:0.38217005133628845
step:66,loss_val:0.3768630027770996
step:71,loss_val:0.372094988822937
step:76,loss_val:0.36777955293655396
step:81,loss_val:0.3638485074043274
step:86,loss_val:0.36024749279022217
step:91,loss_val:0.35693222284317017
step:96,loss_val:0.35386645793914795
Accuracy: 0.9086
"""

