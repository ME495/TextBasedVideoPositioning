import tensorflow as tf
import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)


def sentence_encoder(inputs, sentence_feature_dim):
    '''
    Bidirectional GRU
    Inputs:
        inputs:tensor, shape=(batch_size, sentence_len, vector_dim)
    Returns:
        results:tensor, shape=(batch_size, sentence_feature_dim)
    '''

    #构造前向和后向GRU cell
    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=sentence_feature_dim)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=sentence_feature_dim)

    #建立双向GRU网络
    # hiddens = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=gru_fw_cell, cell_bw=gru_bw_cell, inputs=input_x1, dtype=tf.float32)
    outputs, states=tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell, cell_bw=gru_bw_cell,inputs=inputs,dtype=tf.float32)

    output = tf.reduce_mean(outputs[0]+outputs[1], axis=1)

    return output
