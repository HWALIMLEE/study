import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)

# 데이터 전처리
# CNN은 4차원
x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1)/255
x_test = x_test.reshape(-1,x_test.shape[1],x_test.shape[2],1)/255

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

x = tf.placeholder(tf.float32, shape=[None,28,28,1])
x_img = tf.reshape(x,[-1,28,28,1])
y = tf.placeholder(tf.float32, shape=[None,10])


learning_rate = 0.1
training_epochs = 15
batch_size = 100
total_batch = x_train.shape[0]//batch_size # 60000/100 #600

# x = tf.placeholder(tf.float32, shape=[None,784])
# y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) #dropout과 반대(1-keep_prob만큼 dropout)


w1 = tf.get_variable("w1",shape=[3, 3, 1, 32]) 
print("=============================w1====================")
print(w1) # (3, 3, 1, 32)
# keras Conv2D(output, kernel_size=(3,3),input_shape=(28,28,1))//(3,3)은 kernel_size// 1은 칼라(채널)(x_img = tf.reshape(x,[-1,28,28,1]))에서 1// 32는 output
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
L1 = tf.nn.conv2d(x_img, w1,strides=[1,1,1,1],padding="SAME") # 양 끝에 숫자는 건들지 않음/ 가운데 두개만 건든다
print("=============================L1========================")
print(L1) # (?, 28, 28, 32) /// #padding='VALID'면 (?, 26, 26, 32)
L1 = tf.nn.selu(L1) #activation은 통과시켜주는 애
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #(?, 14, 14, 32) #shape가 반잘려짐
print(L1)

w2 = tf.get_variable("w2",shape=[3, 3, 32, 64])  # 위에서 나온 output 32(상위레이어에 아웃풋을 받아들인다=채널), 여기서 내보내는 output64
print("=============================w1====================")
print(w2) # (3, 3, 1, 32)
# keras Conv2D(output, kernel_size=(3,3),input_shape=(28,28,1))//(3,3)은 kernel_size// 1은 컬럼(x_img = tf.reshape(x,[-1,28,28,1]))에서 1// 32는 output
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
L2 = tf.nn.conv2d(L1, w2,strides=[1,1,1,1],padding="SAME") # 양 끝에 숫자는 건들지 않음/ 가운데 두개만 건든다
print("=============================L1========================")
print(L2) # (?, 28, 28, 32) /// #padding='VALID'면 (?, 26, 26, 32)
L2 = tf.nn.selu(L2) #activation은 통과시켜주는 애
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") 

print(L2) #(?, 7, 7, 64) #maxpooling 2번(2번 반띵)

L2_flat = tf.reshape(L2, [-1,7*7*64]) #(-1,7,7,64)
# 최종 아웃풋
w3 = tf.get_variable("w5",shape=[7*7*64,10],initializer=tf.contrib.layers.xavier_initializer()) # Variable===>get.Variable
# (w1 = tf.Variable(tf.random_normal([784,512]),name='weight'))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L2_flat,w3)+b3)


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # categorical crossentropy 풀어쓴 것

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs): # 15
    avg_cost = 0 
    for i in range(total_batch):    # 600
        batch_xs, batch_ys = x_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size] # wrong
              
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob : 0.7} # 0.7만큼 남기겠다
        c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += c/total_batch  # 여기 무슨 뜻인지 이해 안됨
    print("Epoch:",'%04d' % (epoch + 1),'cost=','{:.9f}'.format(c))
# print("훈련 끗")

prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1)) #boolean값으로 변환(True, False, True,,,)
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))     #[1 0 0 1 1... ]평균값 구함
print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test, keep_prob:0.7})) ##acc출력할 것
