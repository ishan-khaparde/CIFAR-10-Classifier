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
dropout = 0.75

#Input Layer
input_layer = tf.layers.Input(shape=[32,32,3],batch_size=batch_size)

#Convolution Layer 1
conv2d_1 = tf.layers.conv2d(inputs=input_layer,filters = 32, kernel_size=[3,3],padding = 'same',activation=tf.nn.relu)

#Pooling Layer
pool2d_1 = tf.layers.max_pooling2d(inputs=conv2d_1,pool_size=[2,2],strides=2)

#Convolution Layer 2
conv2d_2 = tf.layers.conv2d(inputs = pool2d_1,filters = 64, kernel_size = [3,3],padding='same',activation=tf.nn.relu)