import tensorflow as tf 
import numpy as np 
from tf import layers
from tf.examples.tutorials.cifar10 import cifar10

#hyperparameters
learning_rate = 0.001
num_epochs = 100000
batch_size = 128
display_step = 10

#Network Params
input_size = 1024
num_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, [None,input_size])
y = tf.placeholder(tf.float32,[None,num_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1] padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(x,weights,biases,dropout):
    x = tf.reshape(x,shape=[-1,32,32,3])

    conv_1 = conv2d(x,weights['wc1'],biases['bc1'])

    conv_1 = maxpool2d(conv_1,k = 2)

    conv_2 = conv2d(conv_1,weights['wc2'], biases['bc2'])

    conv_2 = maxpool2d(conv_2,k=2)

    conv_3 = conv2d(conv_2,weights['wc3',biases['bc3']])

    conv_3 = maxpool2d(conv_3,k=2)

    fc1 = tf.reshape(conv_3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1,dropout)

    return tf.add(tf.matmul(fc1,weights['out'],biases['out']))