import unittest
import tensorflow as tf
import sys
import os
test_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
layer_dir = os.path.join(root_dir, 'layers')
sys.path.append(root_dir)
sys.path.append(layer_dir)
import config
from layers import localization_layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

config = config.get_config()

class LocalizationLayerTest(unittest.TestCase):

    def setUp(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=session_config)
        # self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_bi_gru(self):
        T = 100
        inputs = tf.random_normal([1, T, config.video_feature_dim])
        output = localization_layer.bi_gru(inputs)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(output)
        self.assertEqual((1, T, config.localize_rnn_dim*2), result.shape)


if __name__ == '__main__':
    unittest.main()