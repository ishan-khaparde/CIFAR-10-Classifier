import tensorflow as tf 
import numpy as np 
from tf import layers
from tf.examples.tutorials.cifar10 import cifar10

class BuildNN:
    #hyperparameters
    learning_rate = 0.001
    num_epochs = 100000
    batch_size = 128
    display_step = 10

    #Network Params
    input_size = 1024
    num_classes = 10
    dropout = 0.75