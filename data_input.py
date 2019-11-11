import  cv2
import os
import glob
import numpy as np
import random
import gensim
import nltk
import config

config = config.get_config()

last_image_dir = None
last_frames = None
def load_rgb_frames(image_dir):
    global last_image_dir, last_frames
    if image_dir == last_image_dir:
        # print('%s pass'%image_dir)
        return last_frames
    frames = []
    filename_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    filename_list.sort()
    for filename in filename_list:
        img = cv2.imread(filename)
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w, h)
            sc = 1+d/min(w, h)
            img = cv2.resize(img, dsize=(0,0), fx=sc, fy=sc)
        img = (img/255.)*2-1
        frames.append(img)
    frames = np.asarray(frames, dtype=np.float32)
    last_image_dir = image_dir
    last_frames = frames
    return frames


def random_crop(frames, output_size):
    '''Crop the given video sequences (t x h x w x 3) at a random location.
    :param frames: video sequences (t x h x w x 3).
    :param output_size: a square crop (size, size).
    :return: video sequences (t x size x size x 3)
    '''
    t, h, w, c = frames.shape
    th, tw = output_size
    if w == tw and h == th:
        i, j = 0, 0
    else:
        i = random.randint(0, h-th) if h!=th else 0
        j = random.randint(0, w-tw) if w!=tw else 0
    frames = frames[:, i:i+th, j:j+tw, :]
    return frames


def get_wordnet_pos(tag):
    '''获取单词的词性
    :param tag:
    :return:
    '''
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


def sentence2vector(sentence_list):
    '''将句子转化成向量
    :param sentence_list: list
    :return: list,词向量列表
    '''
    # 导入glove模型
    wrod2vec_file = 'word2vec.txt'
    gensim.scripts.glove2word2vec.glove2word2vec(os.path.join(config.glove_dir,'glove.6B.50d.txt'), wrod2vec_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(wrod2vec_file)

    # 把句子转换成词，将大写字母转化成小写并去除标点符号
    takens_list = [gensim.utils.simple_preprocess(sentence, deacc=True) for sentence in sentence_list]

    # 词性标注
    tags_list = [nltk.pos_tag(takens) for takens in takens_list]

    # 词形还原
    wnl = nltk.stem.WordNetLemmatizer()
    words_list = [[wnl.lemmatize(tag[0], pos=get_wordnet_pos(tag[1])) for tag in tags] for tags in tags_list]

    # 过滤掉字典中未出现的词
    words_list = [list(filter(lambda x: x in model.vocab, words)) for words in words_list]

    # 生成词向量
    vectors_list = [[model[word] for word in words] for words in words_list]
    vectors_list = map(np.asarray, vectors_list)
    return vectors_list


def data_gen(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    sentence_list = [line.strip().split('##')[1] for line in lines]
    id_list = [line.strip().split('##')[0].split()[0] for line in lines]
    label_list = [list(map(float, line.strip().split('##')[0].split()[1:])) for line in lines]

    vectors_list = sentence2vector(sentence_list)

    for id, label, vectors in zip(id_list, label_list, vectors_list):
        frames = load_rgb_frames(os.path.join(config.rgb_path, id))
        frames = random_crop(frames, (224, 224))
        label = np.asarray(label, np.float32)
        yield frames, label, np.asarray(vectors)