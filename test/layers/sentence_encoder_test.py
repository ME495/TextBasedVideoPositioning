import tensorflow as tf
import numpy as np
import unittest
import sys
import os
test_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
layer_dir = os.path.join(root_dir, 'layers')
sys.path.append(root_dir)
sys.path.append(layer_dir)
from config import get_config
from sentence_encoder import sentence_encoder

config = get_config()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SentenceEncoderTest(unittest.TestCase):
    def setUp(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=session_config)
        self.training = tf.placeholder(tf.bool)
        # self.sess = tf.Session()


    def tearDown(self):
        self.sess.close()


    def test_sentence_encoder(self):
        input_placeholder = tf.placeholder(dtype=tf.float32,
            shape=(config.batch_size, None, config.word_embedding_dim))
        output = sentence_encoder(input_placeholder, config.sentence_feature_dim)
        self.sess.run(tf.global_variables_initializer())
        sentence_len = 16
        inputs = np.random.rand(config.batch_size, sentence_len, config.word_embedding_dim)
        result = self.sess.run(output, feed_dict={input_placeholder: inputs})
        self.assertEqual((config.batch_size, config.sentence_feature_dim), result.shape)


if __name__ == '__main__':
    unittest.main()