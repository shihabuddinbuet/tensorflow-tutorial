
import tensorflow as tf
from tensorflow.python.framework.dtypes import int32, float32

m = tf.Variable(0.3)
c = tf.Variable(-0.3)
x = 1

y = m*x + c

with tf.Session() as ses:
    ses.run(tf.global_variables_initializer())
    print(ses.run(y))

w = tf.Variable([1], float32)
b = tf.Variable([2], float32)
elements = [1,2,3,4]

w_add = tf.add(w , 1)
b_add = tf.add(b, 2)
w_update = tf.assign(w,w_add)
b_update = tf.assign(b, b_add)

with tf.Session() as ses:
    ses.run(tf.global_variables_initializer())
    for x in elements:
        y = w*x + b
        print("The value of w = %f" % ses.run(w))
        print("The value of x = %f" % x)
        print("The value of b = %f" % ses.run(b))
        print("The value of y = %f" % ses.run(y))
        print("***************Finished round **********************")
        ses.run(w_update)
        ses.run(b_update)