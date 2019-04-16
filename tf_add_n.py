import tensorflow as tf
import numpy as np
input1 = tf.constant([1.0, 2.0, 3.0])
input2 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2])
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(input1))
  print(sess.run(input2))
  print(sess.run(output))
