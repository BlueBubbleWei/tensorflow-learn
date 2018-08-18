# L2正则化
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
BATCH_SIZE = 100
INPUT = 784
OUTPUT = 10
NUM_CHANNELS = 1
IMAGE = 28
L2_REGULARIZER = 0.01
LEARNING_RATE = 0.001
TRAINING_STEPS = 300
MODEL_PATH = 'CONV_Mnist'
MODEL_NAME = 'model.ckpt'
# 初始化权重
weightList = []
# 防止变量名重复
tf.reset_default_graph()


def init_weight(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    weightList.append(tf.nn.l2_loss(weights))
    return weights


# 初始化偏执
def init_bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


# 卷积层封装
def con2d(x, weights, strides, padding_type):
    return tf.nn.conv2d(x, weights, strides=strides, padding=padding_type)


# 池化层
def pooling(x, kernel, strides, padding_type):
    return tf.nn.max_pool(x, ksize=kernel, strides=strides, padding=padding_type)


x = tf.placeholder(tf.float32, shape=[None, INPUT], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT], name='y')
x_ = tf.reshape(x, [-1, IMAGE, IMAGE, NUM_CHANNELS])

# 卷积 # 第一层p=(f-1)/2=2 n+2p-f+1=28+2*2-5+1=28
weights_1 = init_weight([5, 5, 1, 32])
bias = init_bias([32])
layer1_conv = tf.nn.relu(con2d(x_, weights_1, [1, 1, 1, 1], 'SAME') + bias)

# 池化 # 第二层p=(f-1)/2=1 n+2p-f+1=26+2*(1/2)-2+1=26  ==>28
layer2_pool = pooling(layer1_conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

# 全连接 fcn
weight_3 = init_weight([140 * 140 * 32, 10])
bias_3 = init_bias([10])
layer2_pool = tf.reshape(layer2_pool, [-1, 140 * 140 * 32])
y = tf.nn.softmax(tf.matmul(layer2_pool, weight_3) + bias_3)

# 设置损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

loss = tf.add_n(weightList) + cross_entropy
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    xs, ys = mnist.train.next_batch(BATCH_SIZE)

    for i in range(TRAINING_STEPS):
        _, correct_, cross_entropy_ = sess.run([train_step, correct, cross_entropy], feed_dict={x: xs, y_: ys})
        if (cross_entropy_ < 2.38):
            saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME))
