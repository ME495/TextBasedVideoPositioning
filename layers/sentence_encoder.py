
import tensorflow as tf
import numpy as np
from config import get_config

def sentence_encoder(input_x,n_steps):
    '''
    Bidirectional GRU
    Inputs:
        input_x:tensor, shape=(batch_size, n_steps, n_input)
        n_steps:length of input sequence
    Returns:
        hiddens:tensor, shape=(batch_size, sentence_feature_dim)
    '''

    config = get_config()

    #将input_x展开为shape=(batch_size, word_dim)的形式，用于静态RNN网络输入
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)

    #构造前向和后向GRU cell
    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=config.sentence_feature_dim)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=config.sentence_feature_dim)

    #建立双向GRU网络
    hiddens = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=gru_fw_cell, cell_bw=gru_bw_cell, inputs=input_x1, dtype=tf.float32)

    return hiddens
