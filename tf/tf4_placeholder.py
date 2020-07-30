import tensorflow as tf
node1 = tf.constant(3.0) # 상수 : 변하지 않는다, 그냥 프린트 하면 자료형이 프린트 된다
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print("node3:",node3)

# placeholder자체에 값을 넣는 것이 아니고 sess.run할 때 feed_dict로 값을 넣는다. 
# input_느낌==placeholder
sess = tf.Session()
a = tf.placeholder(tf.float32) #placeholder는 sess.run할 때 집어넣는다(input과 비슷한 개념) ===> sess.run에 feed_dict가 들어간다/ placehoder는 출력되는 값 자체는 없다
b = tf.placeholder(tf.float32)

adder_node = a+b
# feed_dict는 '집어 넣는다' 의미
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))  # 7.5
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]})) # [3. 7.] numpy식 연산

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:3,b:4.5})) # 22.5

