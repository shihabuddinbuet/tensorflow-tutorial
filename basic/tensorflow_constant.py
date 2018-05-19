
from __future__ import division
import tensorflow as tf

node1 = tf.constant('Hello world')

with tf.Session() as ses:
    print(ses.run(node1))

a = tf.constant(3)
b = tf.constant(4)

with tf.Session() as ses:
    print("Addition of two nodes(a+b) = %i" % ses.run(a + b))
    print("Subtraction of two nodes(a-b) = %i" % ses.run(a - b))
    print("Multiplication of two nodes(a*b) = %i" % ses.run(a * b))
    print("Division of two nodes(a/b) = %f" % ses.run(a/b))

