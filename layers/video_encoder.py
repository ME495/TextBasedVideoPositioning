import config as cg
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import I3D.i3d as i3d
import sonnet as snt

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 79
_SLIDE_STEP_SIZE = 1  #滑动窗口步长
_CHECKPOINT_PATHS = {
    'rgb': '../I3D/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '../I3D/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '../I3D/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '../I3D/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '../I3D/data/checkpoints/flow_imagenet/model.ckpt',
}
_LABEL_MAP_PATH = '../I3D/data/label_map.txt'
_LABEL_MAP_PATH_600 = '../I3D/data/label_map_600.txt'

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
def GetDataset():
    return np.zeros((8, 90, 224, 224, 3))
def GetRGBFeature(rgb_sample,times,rgb_input,rgb_feature,rgb_saver):
    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        #rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
        feed_dict[rgb_input] = rgb_sample
        out_rgb_feature = sess.run(rgb_feature,feed_dict=feed_dict)
    return out_rgb_feature

def video_encoder():
    """
    Video encoder.
    Inputs:

    Return:
        tensor, shape=(batch_size, sentence_feature_dim).
    """

    Video = GetDataset()
    T = Video.shape[1]-_SAMPLE_VIDEO_FRAMES+_SLIDE_STEP_SIZE
    tf.logging.set_verbosity(tf.logging.INFO)
    #imagenet_pretrained = FLAGS.imagenet_pretrained
    NUM_CLASSES = 400
    #kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(  #定义rgb流输入
        tf.float32,
        shape=(8, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(  #定义rgb模型
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
      rgb_feature, rgb_model_output = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
      rgb_feature = tf.nn.avg_pool3d(rgb_feature, ksize=[1, 10, 7, 7, 1],  #卷积核的大小第二维不大确定
                           strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      rgb_feature = tf.reshape(rgb_feature,(8,1,1024))
    rgb_variable_map = {}
    for variable in tf.global_variables():
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    for i in range(Video.shape[1]-_SAMPLE_VIDEO_FRAMES+_SLIDE_STEP_SIZE):
        if i==0:
            a = Video[:,i:i+_SAMPLE_VIDEO_FRAMES,:,:,:]
            output = GetRGBFeature(Video[:,i:i+_SAMPLE_VIDEO_FRAMES,:,:,:],i,rgb_input,rgb_feature,rgb_saver)
        else:
            output = tf.concat([output,GetRGBFeature(Video[:,i:i+_SAMPLE_VIDEO_FRAMES,:,:,:],i,rgb_input,rgb_feature,rgb_saver)],axis=1)
        print(i)


video_encoder()