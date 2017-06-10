import tensorflow as tf


coty = tf.Variable(15, tf.float32)

init = tf.global_variables_initializer()
session = tf.Session()


print('coty = ', session.run(coty))
x = tf.get_variable('coty', 1, dtype=tf.float64)

session.run(init)

print('x = ', session.run(x))





