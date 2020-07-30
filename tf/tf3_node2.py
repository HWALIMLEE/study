# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2

# node
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)
node4 = tf.constant(2.0)

# 덧셈
sum = tf.add_n([node1, node2, node3])
sess = tf.Session()
sum_ = sess.run(sum)
print(sum_)

# 뺼셈
diff = tf.subtract(node2, node1)
diff_ = sess.run(diff)
print(diff_)

# 곱셈
mult = tf.multiply(node1, node2)
mult_ = sess.run(mult)
print(mult_)

# 나눗셈
div = tf.divide(node2,node4)
div_ = sess.run(div)
print(div_)