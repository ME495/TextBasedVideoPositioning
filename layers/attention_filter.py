
import tensorflow as tf
import os
import sys
import numpy as np
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)
import config

config = config.get_config()


def attention_filter(video_feature, sentence_feature):
    '''
    Attention.
    Inputs:
        video_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
        sentence_feature: tensor, shape=(batch_size, sentence_feature_dim).
    Returns:
    A tensor, shape=(batch_size, T, video_feature_dim).
    '''
    video_feature_shape = video_feature.get_shape()
    batch_size, T, dim = video_feature_shape
    theta_sentence_feature = tf.layers.dense(sentence_feature, config.video_feature_dim, activation=tf.nn.tanh)
    matmul = tf.matmul(video_feature, tf.reshape(theta_sentence_feature, (batch_size, dim, 1)))
    A = tf.nn.softmax(tf.reshape(matmul, (batch_size, -1)))
    result = video_feature*tf.tile(tf.reshape(A, (batch_size, T, 1)), [1, 1, config.video_feature_dim])
    return result
