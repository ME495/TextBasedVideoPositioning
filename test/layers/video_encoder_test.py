import unittest
import  tensorflow as tf
import numpy as np
import sys
import os
test_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
layer_dir = os.path.join(root_dir, 'layers')
sys.path.append(root_dir)
sys.path.append(layer_dir)
from layers.video_encoder import video_encoder
import config

config = config.get_config()

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class VideoEncoderTest(unittest.TestCase):

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


    def test_video_encoder(self):

        videos_placeholder = tf.placeholder(tf.float32,
            shape=(config.batch_size, None, config.image_size, config.image_size, config.image_channel))
        video_features = video_encoder(
            videos_placeholder, config.sample_len, config.slide_step_size)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            # print(variable.name)
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(":0", "")] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(self.sess, config.i3d_checkpoint['rgb'])
        self.sess.run(tf.global_variables_initializer())
        video_len = 90
        videos = np.random.random(
            [config.batch_size, video_len, config.image_size, config.image_size, config.image_channel])
        # print(videos.shape)
        result = self.sess.run(video_features, feed_dict={videos_placeholder: videos})
        self.assertEqual((config.batch_size, (video_len-config.sample_len+1)//config.slide_step_size, config.video_feature_dim), result.shape)

        video_len = 120
        videos = np.random.random(
            [config.batch_size, video_len, config.image_size, config.image_size, config.image_channel])
        # print(videos.shape)
        result = self.sess.run(video_features, feed_dict={videos_placeholder: videos})
        self.assertEqual((config.batch_size, (video_len - config.sample_len + 1) // config.slide_step_size,
                          config.video_feature_dim), result.shape)



if __name__ == '__main__':
    unittest.main()
