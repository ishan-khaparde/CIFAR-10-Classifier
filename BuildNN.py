import tensorflow as tf 
import numpy as np 
from tf import layers
from tensorflow.models.image.cifar10 import cifar10

#hyperparameters
learning_rate = 0.001
num_epochs = 100000
batch_size = 128
display_step = 10

#Network Params
input_size = 1024
num_classes = 10
dropout = 0.4
mode = tf.estimator.ModeKeys.TRAIN
#Input Layer
input_layer = tf.layers.Input(shape=[32,32,3],batch_size=batch_size)

#Convolution Layer 1
conv2d_1 = tf.layers.conv2d(inputs=input_layer,filters = 32, kernel_size=[3,3],padding = 'same',activation=tf.nn.relu)

#Pooling Layer 1
pool2d_1 = tf.layers.max_pooling2d(inputs=conv2d_1,pool_size=[2,2],strides=2)

#Convolution Layer 2
conv2d_2 = tf.layers.conv2d(inputs = pool2d_1,filters = 64, kernel_size = [3,3],padding='same',activation=tf.nn.relu)

#Pooling layer 2
pool2d_2 = tf.layers.max_pooling2d(inputs = conv2d_2,pool2d_2=[2,2]),strides = 2)

#Dense Layer
flatten = tf.reshape(pool2d_2,[-1,8,8,64])
dense_layer = tf.layers.dense(inputs=flatten,units=1024,activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs = dense_layer,rate = dropout,training=mode == tf.estimator.ModeKeys.TRAIN)

#Output Layer
output = tf.layers.dense(inputs = dropout,units = 10)

#Collect the results
results = {
    'predicted' : tf.argmax(output,axis = 1)
    'score' : tf.nn.softmax(output,name = "softmax_tensor") 
}

if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,predictions = results)

labels_predicted = tf.one_hot

