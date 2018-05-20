import tensorflow as tf
from tensorflow.python.framework.dtypes import float32

w = tf.Variable([0.3], float32)
c = tf.Variable([-0.3], float32)
feature = tf.placeholder(float32)

y_est = w*feature + c
y_actual = tf.placeholder(float32)
loss_func = tf.reduce_sum(tf.square(y_actual - y_est))

# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print("loss after training the function : %f "
#           % sess.run(loss_func, feed_dict={feature:[1,2,3,4], y_actual:[-1, -2, -3, -4]}))
#
#     print("final weight after training : %f " % sess.run(w))
#     print("constant value after training : %f" % sess.run(c))

num_iteration = 100
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss_func)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_iteration):
        sess.run(train, feed_dict={feature:[1,2,3,4], y_actual:[-1, -2, -3, -4]})

    print(sess.run([w,c]))
    print(sess.run(loss_func,feed_dict={feature:[1,2,3,4], y_actual:[-1, -2, -3, -4]}))








