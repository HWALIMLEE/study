import tensorflow as tf
tf.set_random_seed(777) # randomseed로 고정


# 변수// 0~1 사이의 정규확률분포 값을 생성해주는 함수
W = tf.Variable(tf.random_normal([1]),name='weight') #2로 바꾸니 에러가 난다. Dimensions must be equal, but are 3 and 2 for 'mul' (op: 'Mul') with input shapes: [3], [2]
b = tf.Variable(tf.random_normal([1]),name='bias')

print(W)

"""올바른 예제"""
# 변수는 feed_dict할 필요 없다
W = tf.Variable([0.3],tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #변수 선언==변수 초기화 꼭 해주어야 한다
aaa = sess.run(W)
print("aaa:",aaa)
sess.close()

"""틀린 예제"""
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #변수 선언==변수 초기화 꼭 해주어야 한다
W = tf.Variable([0.3],tf.float32)
print(sess.run(W))
sess.close()
"""
# 변수 먼저 선언해줄 것

"""InteractiveSession을 쓰게 되면 .eval 쓰면 된다"""
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print("bbb:",bbb)
sess.close()

"""Session에서도 eval먹힌다"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc:",ccc)
sess.close()

