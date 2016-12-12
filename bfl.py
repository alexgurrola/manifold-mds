import tensorflow as tf
import numpy as np

# Generate pseudo-random data
x_data = np.random.rand(100).astype(np.float32)
y_data = (x_data * 0.1) + 0.3

# Define vectors
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = (W * x_data) + b

# Minimize mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Create session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Calibrate session
W_prev, b_prev = sess.run(W), sess.run(b)

# Train according to parameters
for step in range(4001):
    sess.run(train)
    if step % 20 == 0:
        if sess.run(W) == W_prev and sess.run(b) == b_prev:
            break
        else:
            print(step, sess.run(W), sess.run(b))
            W_prev, b_prev = sess.run(W), sess.run(b)

# Results
print('BFL', sess.run(W), sess.run(b))