import tensorflow as tf

# MNIST Data Sets
print("Prepare Environment\n")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# SoftMax Regression Tensors
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training Calibration
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Predictive Calibration
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_accuracy = 0

# Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Label Algorithm
print("\nCross-Entropy Gradient Descent Optimizer\n")

# Train
for step in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if step % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("step %d, training accuracy %g" % (step, train_accuracy))
        if train_accuracy > 0.92:
            break
