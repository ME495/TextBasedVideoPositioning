
class Config(object):
    data_path = '/data02/chengjian19/dataset/Charades/'
    raw_video_path = '/data02/chengjian19/dataset/Charades/Charades_v1/'
    video480_path = '/data02/chengjian19/dataset/Charades/Charades_v1_480/'
    rgb_path = '/data02/chengjian19/dataset/Charades/Charades_v1_rgb/'
    flow_path = '/data02/chengjian19/dataset/Charades/Charades_v1_flow/'
    train_file = '/data02/chengjian19/dataset/Charades/charades_sta_train.txt'
    test_file = '/data02/chengjian19/dataset/Charades/charades_sta_test.txt'

    video_feature_dim = 1024
    sentence_feature_dim = 300
    localize_rnn_dim = 512
    hidden_dim = 256


def get_config():
    config = Config()
    
    # config.data_path = '/data02/chengjian19/dataset/Charades/'
    # config.raw_video_path = '/data02/chengjian19/dataset/Charades/Charades_v1/'
    # config.video480_path ='/data02/chengjian19/dataset/Charades/Charades_v1_480/'
    # config.rgb_path = '/data02/chengjian19/dataset/Charades/Charades_v1_rgb/'
    # config.flow_path = '/data02/chengjian19/dataset/Charades/Charades_v1_flow/'
    # config.train_file = '/data02/chengjian19/dataset/Charades/charades_sta_train.txt'
    # config.test_file = '/data02/chengjian19/dataset/Charades/charades_sta_test.txt'
    #
    # config.video_feature_dim = 1024
    # config.sentence_feature_dim = 300
    
    return config
    