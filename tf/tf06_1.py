import tensorflow as tf
tf.set_random_seed(777) # randomseed로 고정

# sess = tf.Session()
x_train = tf.placeholder(tf.float32, shape=[None])  # placeholder는 sess.run할 때 집어넣는다(input과 비슷한 개념) ===> sess.run에 feed_dict가 들어간다/ placehoder는 출력되는 값 자체는 없다
y_train = tf.placeholder(tf.float32, shape=[None])

# feed_dict는 '집어 넣는다' 의미
# print(sess.run(adder_node, feed_dict={a:3, b:4.5}))  # 7.5
# print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]})) # [3. 7.] numpy식 연산

# add_and_triple = adder_node * 3
# print(sess.run(add_and_triple, feed_dict={a:3,b:4.5})) # 22.5


# 변수// 0~1 사이의 정규확률분포 값을 생성해주는 함수
W = tf.Variable(tf.random_normal([1]),name='weight') #2로 바꾸니 에러가 난다. Dimensions must be equal, but are 3 and 2 for 'mul' (op: 'Mul') with input shapes: [3], [2]
b = tf.Variable(tf.random_normal([1]),name='bias')


# sess = tf.Session()
# 변수는 항상 초기화 시키고 작업해야한다. 이거 안하면 에러 뜬다
# sess.run(tf.global_variables_initializer())
# print(sess.run(W))

hypothesis = x_train * W + b

sess = tf.Session()

cost = tf.reduce_mean(tf.square(hypothesis-y_train)) # MSE

# cost = loss
# reduce_mean = 평균
 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # optimizer// 최소의 loss가 최적의 weight값 구해줌

# Session범위 안에 다 들어감
# cost, weight, bias 구했음
# hypothesis에 새로운 값 넣어주기..?
with tf.Session() as sess: #tf.Session 보다는 tf.compat.v1.Session(version문제)
    sess.run(tf.global_variables_initializer())
    # sess.run(feed_dict={x_train:[1,2,3],y_train:[3,5,7]})==>error
    # 전체 변수 싹 초기화(with문에서 한방에 초기화)==변수 선언, with 문에 한번
    # with문은 session close해주기 위해 쓰는 것 뿐, with 문은 sess close해주는 거 명시 안해줘도 되기 때문에 써준 것 뿐
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],feed_dict={x_train:[1,2,3],y_train:[3,5,7]}) # train은 하되 결과값은 출력하지 않겠다.// cost를 minimize

        if step % 20 ==0:
            print(step, cost_val, W_val, b_val)

# session 열었으면 닫아주어야 함
# cost값을 계속 최소화시켜주는 것===> weight값을 계속 반영시킴

# predict해보기
# 4라는 새로운 값 넣어서 새로운 hypothesis값 예측해보기
    print("예측:",sess.run(hypothesis, feed_dict={x_train:[4]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))


