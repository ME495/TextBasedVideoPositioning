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

NUM_CHANNELS = 1
CONV1_WIDTH = 1
CONV1_HEIGHT = 1
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
DIM_OUT = video_feature_dim


def get_weight(shape, regularizer):
    #获得某一层的w值，使用L2正则化
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    #获得某一层的偏置，初值全零
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')


def get_conv_w_b(regularizer):
    global CONV1_WIDTH, CONV1_HEIGHT, NUM_CHANNELS, CONV1_KERNEL_NUM
    conv_w = get_weight([CONV1_WIDTH, CONV1_HEIGHT, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv_b = get_bias([CONV1_KERNEL_NUM])
    return conv_w, conv_b


def g_mul(G, h, regularizer):
    global DIM_OUT
    G_shape = tf.shape(G)
    with tf.Session() as sess:
        dim_t = sess.run(G_shape)[1]
        G = tf.reshape(G, [dim_t, DIM_OUT])
        h = tf.reshape(h, [1, DIM_OUT])
        A0 = tf.reduce_sum(tf.multiply(G, h), reduction_indices=1)
        A = tf.nn.softmax(tf.multiply(tf.constant([0.03125]), A0))
        G_mean = tf.multiply(tf.reshape(a, [-1,1]), G)
    return G_mean


def process_sentence(sentence_feature, regularizer):
    global DIM_OUT, sentence_feature_dim
    convs = get_conv_w_b(regularizer)
    convs_w, convs_b = convs[0], convs[1]
    with tf.Session() as sess:
        sentence_shape = sess.run(tf.shape(sentence_feature))
        BATCH_SIZE = sentence_shape[0]
        sentence_feature = tf.reshape(sentence_feature, [BATCH_SIZE, 1, sentence_feature_dim, 1])
    conv1 = conv2d(sentence_feature, convs_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, convs_b))
    relu_shape = relu1.get_shape().as_list()
    nodes = sentence_feature_dim
    reshaped = tf.reshape(relu1, [relu_shape[0], nodes])
    fcl_w = get_weight([nodes, DIM_OUT], regularizer)
    fcl_b = get_bias([DIM_OUT])
    fcl = tf.nn.relu(tf.matmul(reshaped, fcl_w) + fcl_b)
    return fc1


def process_video(video_feature, regularizer):
    global video_feature_dim
    convv = get_conv_w_b(regularizer)
    convv_w, convv_b = convv[0], convv[1]
    with tf.Session() as sess:
        video_shape = sess.run(tf.shape(video_feature))
        batch_size, dim_t = video_shape[0], video_shape[1]
        video_feature = tf.reshape(video_feature, [batch_size, dim_t, video_feature_dim, 1])
    conv1 = conv2d(video_feature, convv_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, convv_b))
    return relu1




def attention_filter(video_feature, sentence_feature, regularizer):
    '''
       Attention.
       Inputs:
           video_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
           sentence_feature: tensor, shape=(batch_size, sentence_feature_dim).
       Returns:
           A tensor, shape=(batch_size, T, video_feature_dim).
    '''
    # 起主要作用的函数
    global video_feature_dim
    sentence = process_sentence(sentence_feature, regularizer)
    video = process_video(video_feature, regularizer)
    G = g_mul(video, sentence, regularizer)
    with tf.Session() as sess:
        G_shape = sess.run(tf.shape(G))
        dim_t = G_shape[0]
    G = tf.reshape(G, [1, dim_t, video_feature_dim])
    return G



