
from __future__ import division
import tensorflow as tf
from tensorflow.python.framework.dtypes import float32, string

node1 = tf.placeholder(string)

with tf.Session() as sess:
    print(sess.run(node1, feed_dict={node1 : "Hello world"}))

a = tf.placeholder(float32)
b = tf.placeholder(float32)

sum = tf.add(a,b)
sub = tf.sub(a,b)
mul = tf.mul(a,b)
div = tf.div(a,b)

with tf.Session() as sess:
    print("Addition of two nodes(a+b) = %f" % sess.run(sum,feed_dict={a:3.0, b:4.0}))
    print("Subtraction of two nodes(a-b) = %f" % sess.run(sub, feed_dict={a:3.0, b:4.0}))
    print("Multiplication of two nodes(a*b) = %f" % sess.run(mul, feed_dict={a:3, b:4}))
    print("Division of two nodes(a/b) = %f" % sess.run(div, feed_dict={a:3, b:4}))



