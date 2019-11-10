import numpy as np
import tensorflow as tf
import i3d
import sonnet as snt
import os
import sys
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)

def video_encoder(video_inputs, sample_len, slide_step_size):
    """
    Video encoder.
    Inputs:
        video_inputs: tensor, shape=(batch_size, sample_len, image_size, image_size, channel), sample_len is variable.
        sample_len: scale value.
        slide_step_size: scale value.
    Return:
        tensor, shape=(batch_size, 1024), The output dim of I3D model is 1024.
    """
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(spatial_squeeze=True, final_endpoint='Mixed_5c')
    # batch_size, depth, _, _, _ = tf.shape(video_inputs)
    depth = tf.shape(video_inputs)[1]

    T = depth-sample_len+1

    def body(step, video_features):
        batch_sample = video_inputs[:, step:step+sample_len]
        batch_sample_features, _ = rgb_model(batch_sample, is_training=False, dropout_keep_prob=1.0)
        batch_sample_features = tf.nn.avg_pool3d(batch_sample_features, ksize=[1, 1, 7, 7, 1],
            strides=[1, 1, 1, 1, 1], padding='VALID')
        batch_sample_features = tf.reduce_mean(batch_sample_features, axis=1)
        batch_sample_features = tf.reshape(batch_sample_features, [-1, 1024])
        video_features = video_features.write(step, batch_sample_features)
        return step+slide_step_size, video_features

    def cond(step, video_features):
        return step+sample_len <= T

    step = 0
    video_features = tf.TensorArray(dtype=tf.float32, size=T)
    step, video_features = tf.while_loop(cond=cond, body=body, loop_vars=(step, video_features))
    video_features = tf.transpose(video_features.stack(), [1, 0, 2])
    return video_features