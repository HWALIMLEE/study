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
# with문 써주든가
# 아니면 sess.close()해주든가
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])


learning_rate = 0.01
training_epochs = 15
batch_size = 100
total_batch = x_train.shape[0]//batch_size # 60000/100 #600

# x = tf.placeholder(tf.float32, shape=[None,784])
# y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) #dropout과 반대(1-keep_prob만큼 dropout)


w1 = tf.get_variable("w1",shape=[784,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
print("=============================w1===============================")
print(w1) #(784, 512)
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b1 = tf.Variable(tf.random_normal([512]))
print("=============================b1============================")
print(b1) #(512, )
L1 = tf.nn.selu(tf.matmul(x,w1) +b1)
print("============================Selu============================")
print(L1) #(None,512)
print("===========================Drop============================")
L1 = tf.nn.dropout(L1,keep_prob = keep_prob)
print(L1) #(None, 512)

w2 = tf.get_variable("w2",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2,keep_prob = keep_prob)

w3 = tf.get_variable("w3",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b2)
L3 = tf.nn.dropout(L3,keep_prob = keep_prob)

w4 = tf.get_variable("w4",shape=[512,256],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4,keep_prob = keep_prob)

w5 = tf.get_variable("w5",shape=[256,10],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4,w5)+b5)


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # categorical crossentropy 풀어쓴 것

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs): # 15
    avg_cost = 0 
    for i in range(total_batch):    # 600
        batch_xs, batch_ys = x_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size] # wrong
        # batch_xs, batch_ys = x_train[0:100], y_train[0:100]
        # batch_xs, batch_ys = x_train[100:200], y_train[100:200]
        # ....==>반복분 써주기
        # batch_xs, batch_ys = x_train[i:batch_size]
        # batch_xs, batch_ys = x_train[i*batch_size:i*batch_size+batch_size]

        #############################
        # start = i * batch_size
        # end = start + batch_size
        # batch_xs , batch_ys = x_train[start:end],y_train[start:end]
        #############################
        
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob : 0.7} # 0.7만큼 남기겠다
        c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += c/total_batch  # 여기 무슨 뜻인지 이해 안됨
    print("Epoch:",'%04d' % (epoch + 1),'cost=','{:.9f}'.format(c))
# print("훈련 끗")

prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1)) #boolean값으로 변환(True, False, True,,,)
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))     #[1 0 0 1 1... ]평균값 구함
print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test, keep_prob:0.7})) ##acc출력할 것


# batch_size 안 쓰면
# 한방에 60,000개씩 돌리는 상황
# 60,000개의 데이터를 100개씩 잘라서 훈련
# 한 epoch에 600번 돈다
# epoch는 15번
# 총 9000번 훈련 ( 600 * 15 = 9000)
# 잘게 잘라서 batch_size훈련하면 9000번 훈련하는 것이 되는 것


########################################################################
# Myung Code
# def next_batch(num, data, labels):
#        '''
#        Return a total of `num` random samples and labels. 
#        '''
#        idx = np.arange(0 , len(data))
#        np.random.shuffle(idx)
#        idx = idx[:num]
#        data_shuffle = [data[i] for i in idx]
#        labels_shuffle = [labels[i] for i in idx]

#        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# batch_xs, batch_ys = next_batch(100, x_train, y_train)
###########################################################################


# 0~99까지만 구성하면 된다// y는 특별히 잘릴필요 없다...?

# [0~100]
# [100~200]

###############keras################### 질문(잘 이해 안감 아직)
# 가중치 연산은 dropout적용 안된다
# 훈련한 가중치를 layer상에서는 dropout먹히지만
# 평가할 때는 dropout먹히지 않는다
# 결과값에 대한 과적합의 원인
#######################################