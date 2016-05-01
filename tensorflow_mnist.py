"""Tensorflow MNIST softmax gradient descent example"""


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def main():
    # load data
    print("Load data")
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # create tensorflow session
    sess = tf.InteractiveSession()

    # creating nodes for the input images and target output classes
    print("Create model")
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # define the weights W and biases b for our model
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    # initialize variables
    sess.run(tf.initialize_all_variables())
    # implement regression model
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # gradient descent training with minibatches
    print("Traning..")
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print("test accuracy %g"%accuracy.eval(feed_dict={    x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    main()


