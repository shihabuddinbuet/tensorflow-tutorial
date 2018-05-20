
import tensorflow as tf
from tensorflow.python.framework.dtypes import float32

w = tf.Variable([0.3])
b = tf.Variable([-0.3])
x = tf.placeholder(float32)
y = tf.placeholder(float32)

linear_model = w*x + b
pred = tf.sigmoid(linear_model)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, targets=y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
       sess.run(train, feed_dict={x:[-1, -2, -3, -4, 1, 2, 3, 4],
                                  y:[0, 0, 0, 0, 1, 1, 1, 1]})

    print(sess.run([w,b]))
    print(sess.run(loss, feed_dict={x:[-1, -2, -3, -4, 1, 2, 3, 4],
                                    y:[0, 0, 0, 0, 1, 1, 1, 1]}))