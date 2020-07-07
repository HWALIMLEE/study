import tensorflow as tf


node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0) #float로 생각할 것
node3 = tf.add(node1, node2)

print("node1:",node1, "node2:",node2)
print("node3:",node3) # 자료형이 나온다 // #input 한 머신의 상태만 나온다. node3: Tensor("Add:0", shape=(), dtype=float32)(Add형)

sess = tf.Session()
print("sess.run(node1, node2):",sess.run([node1,node2]))  # [3.0, 4.0]
print(sess.run(node3))   # 7.0

