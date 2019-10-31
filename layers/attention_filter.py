
def attention_filter(video_feature, sentence_feature):
    '''
    Attention.
    Inputs:
        video_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
        sentence_feature: tensor, shape=(batch_size, sentence_feature_dim).
    Returns:
        A tensor, shape=(batch_size, T, video_feature_dim).
    '''
    #起主要作用的函数
    object0 = config.get_config()
    video_feature_dim = object0.video_feature_dim
    sentence_feature_dim = object0.sentence_feature_dim
    return()
'''
Attention.
Inputs:
    video_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
    sentence_feature: tensor, shape=(batch_size, sentence_feature_dim).
Returns:
    A tensor, shape=(batch_size, T, video_feature_dim).
'''
    pass