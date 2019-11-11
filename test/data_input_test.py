import unittest
import sys
import os
import tensorflow as tf
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_dir)
import config
import data_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

config = config.get_config()


class DataInputTest(unittest.TestCase):

    # @unittest.skip('test_load_rgb_frames.')
    def test_load_rgb_frames(self):
        dir = os.path.join(config.rgb_path, 'XJA7Z')
        frames = data_input.load_rgb_frames(dir)
        for frame in frames:
            self.assertGreaterEqual(226, min(frame.shape[1:3]))

    # @unittest.skip('test_random_crop.')
    def test_random_crop(self):
        dir = os.path.join(config.rgb_path, 'QE4YE')
        frames = data_input.load_rgb_frames(dir)
        frames = data_input.random_crop(frames, (224, 224))
        self.assertEqual((224, 224,), frames.shape[1:3])


    def test_data_gen(self):
        # data_input.data_gen(config.test_file)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        dataset = tf.data.Dataset.from_generator(
            lambda: data_input.data_gen(config.test_file),
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([2]), tf.TensorShape([None, 50])))
        dataset = dataset.shuffle(buffer_size=3*config.batch_size).padded_batch(
            8, padded_shapes=([None, 224, 224, 3], [None], [None, 50]))
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session(config=session_config) as sess:
            videos, labels, vectors = sess.run(next_element)
            self.assertEqual((224, 224, 3), videos.shape[2:])
            self.assertEqual(8, videos.shape[0])
            self.assertEqual((8, 2), labels.shape)
            self.assertEqual(8, vectors.shape[0])
            self.assertEqual(50, vectors.shape[2])


if __name__ == '__main__':
    unittest.main()
