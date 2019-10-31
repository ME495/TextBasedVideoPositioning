import tensorflow as tf
import os
import sys
import numpy as np
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)
import config

object0 = config.get_config()
video_feature_dim = object0.video_feature_dim
sentence_feature_dim = object0.sentence_feature_dim

IMAGE_SIZE = video_feature_dim
NUM_CHANNELS = 3
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 300


def get_weight(shape, regularizer):
    #获得某一层的w值，使用L2正则化
    w = tf.Variable(tf.truncated_normal(shape,stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return(w)


def get_bias(shape):
    #获得某一层的偏置，初值全零
    b = tf.Variable(tf.zeros(shape))
    return(b)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def theta(x,regularizer):
    w1 = get_weight([IMAGE_SIZE, OUTPUT_NODE], regularizer)
    b1 = get_bias([OUTPUT_NODE])
    y = tf.nn.relu(tf.matmul(x, w1) + b1)
    return(y)


def A(G,h):
    GT = np.array(G).reshape(-1,1)
    theta_h = theta(h,0.001)
    y = tf.nn.softmax(np.dot(GT,theta_h)/18)
    return(y)


def G_mean(A,G):
    G_m = np.multiply(A,G)
    return(G_m)


def attention_filter(video_feature, sentence_feature):

    '''
    Attention.
    Inputs:
        video_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
        sentence_feature: tensor, shape=(batch_size, sentence_feature_dim).
    Returns:
        A tensor, shape=(batch_size, T, video_feature_dim).
    '''
    #起主要作用的函数

    return()
