import tensorflow as tf
print(tf.__version__)

# constant = 상수
hello = tf.constant("Hello world")

print(hello) # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello)) # b'Hello world' # bite

# 값을 눈에 보이게 하고 싶다면 session을 써야 한다 
# 이 session절차를 없앤 것이 keras
# 3차원 이상부터는 tensor
# 1차원 벡터, 2차원 행렬, 3차원 tensor

