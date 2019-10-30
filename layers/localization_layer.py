
def localication_layer(attention_feature):
    '''
    Locatication layer.
    Inputs:
        attention_feature: tensor, shape=(batch_size, T, video_feature_dim), T is variable.
    Returns:
        A tensor, shape=(batch_size, T).
    '''
