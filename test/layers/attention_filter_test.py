import unittest
import  tensorflow as tf
import sys
import os
test_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
sys.path.append(root_dir)
from layers import attention_filter
import config

config = config.get_config()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class AttentionFilterTest(unittest.TestCase):

    def setUp(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=session_config)


    def tearDown(self):
        self.sess.close()


    def test_attention_layer(self):
        T = 100
        batch_size = 1
        video_feature = tf.random.normal([batch_size, T, config.video_feature_dim])
        sentence_feature = tf.random.normal([batch_size, config.sentence_feature_dim])
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        output = attention_filter.attention_filter(video_feature, sentence_feature)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(output)
        # self.assertEqual((batch_size, T, config.video_feature_dim), result.shape)


if __name__ == '__main__':
    unittest.main()
