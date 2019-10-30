import tensorflow as tf
import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)
import config

config = config.get_config()

def bi_gru(inputs):
    '''
    Bidirectional GRU.
    Inputs:
        inputs: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
    Returns:
        output: tensor, shape(batch_size, T, fc_dim), T is variable.
    '''
    gru_fw1 = tf.nn.rnn_cell.GRUCell(num_units=config.localize_rnn_dim)
    gru_forward = tf.nn.rnn_cell.MultiRNNCell(cells=[gru_fw1])

    gru_bw1 = tf.nn.rnn_cell.GRUCell(num_units=config.localize_rnn_dim)
    gru_backward = tf.nn.rnn_cell.MultiRNNCell(cells=[gru_bw1])

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=gru_forward, cell_bw=gru_backward, inputs=inputs, dtype=tf.float32)
    output = tf.concat(outputs, 2)
    return output


def localication_layer(attention_feature):
    '''
    Locatication layer.
    Inputs:
        attention_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
    Returns:
        A tensor, shape=(batch_size, T).
    '''
    pass
