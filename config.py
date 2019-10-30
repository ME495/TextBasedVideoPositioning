
class Config(object):
    pass

def get_config():
    config = Config()
    
    config.data_path = '/data02/chengjian19/dataset/Charades/'
    config.raw_video_path = '/data02/chengjian19/dataset/Charades/Charades_v1/'
    config.video480_path ='/data02/chengjian19/dataset/Charades/Charades_v1_480/'
    config.rgb_path = '/data02/chengjian19/dataset/Charades/Charades_v1_rgb/'
    config.flow_path = '/data02/chengjian19/dataset/Charades/Charades_v1_flow/'
    config.train_file = '/data02/chengjian19/dataset/Charades/charades_sta_train.txt'
    config.test_file = '/data02/chengjian19/dataset/Charades/charades_sta_test.txt'
    
    config.video_feature_dim = 1024
    config.sentence_feature_dim = None
    
    return config
    