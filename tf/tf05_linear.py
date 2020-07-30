import tensorflow as tf
tf.set_random_seed(777) # randomseed로 고정

x_train = [1,2,3]
y_train = [3,5,7]

# 변수// 0~1 사이의 정규확률분포 값을 생성해주는 함수
W = tf.Variable(tf.random_normal([1]),name='weight') #2로 바꾸니 에러가 난다. Dimensions must be equal, but are 3 and 2 for 'mul' (op: 'Mul') with input shapes: [3], [2]
b = tf.Variable(tf.random_normal([1]),name='bias')

# sess = tf.Session()
# 변수는 항상 초기화 시키고 작업해야한다. 이거 안하면 에러 뜬다
# sess.run(tf.global_variables_initializer())
# print(sess.run(W))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # MSE
# cost = loss
# reduce_mean = 평균
 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # optimizer// 최소의 loss가 최적의 weight값 구해줌

#Session범위 안에 다 들어감
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 전체 변수 싹 초기화

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # train은 하되 결과값은 출력하지 않겠다.// cost를 minimize

        if step % 20 ==0:
            print(step, cost_val, W_val, b_val)

# session 열었으면 닫아주어야 함
