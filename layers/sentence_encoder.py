
def sentence_encoder():
    '''
    Sentence encoder.
    Inputs:

import tensorflow as tf
import numpy as np
from config import get_config

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

import random

#全局变量，用于batch加载
t = 0

def get_input():
    #将glove模型加载到model中

    # 输入文件
    glove_file = "./glove.6B/glove.6B.100d.txt"
    # 输出文件
    tmp_file = "./glove.6B/test_word2vec.txt"

    # 将glove模型转化为word2vec的形式
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)

    # 加载转化后的文件
    model = KeyedVectors.load_word2vec_format(tmp_file)
    #print(model['the']) 
    #print(model.wv.syn0[1])
    return model

def get_batch_x(model,batch_size):
    '''
    Inputs: 
        model:词向量模型
        batch_size:batch的大小
    Outputs:
        batch_x: shape(batch_size,n_inputs)
    '''

    batch_x=[]
    global t

     #重复使用数据集，此可改为mod的形式
    if ((t+1)*batch_size > 400001):
        t = 0

    #读入一个batch的x
    for i in range(t*batch_size, (t+1)*batch_size):
        batch_x.append(model.wv.syn0[i])

    #list型报错？ 转化为array型
    batch_x = np.array(batch_x)
    t = t+1
    return batch_x

def get_batch_y(batch_size):
    '''
    Inputs: 
        batch_size:batch的大小
    Outputs:
        batch_Y: shape(batch_size,n_classes)
    '''

    batch_y=[]
    #方便单元测试，随便设定的y
    for i in range(0, batch_size):
        batch_y.append(i);
    batch_y = np.array(batch_y)
    return batch_y

def single_layer_static_bi_gru(input_x,n_steps,n_hidden):

    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)

    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)

    hiddens,fw_state,bw_state = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=gru_fw_cell,cell_bw=gru_bw_cell,inputs=input_x1,dtype=tf.float32)

    print('hiddens:\n',type(hiddens),len(hiddens),hiddens[0].shape)
    #print('hiddens:\n',type(hiddens),len(hiddens),hiddens[1].shape)

    print(hiddens)

    return hiddens,fw_state,bw_state

def  mnist_rnn_classfication():
    tf.reset_default_graph()

    config = get_config()

    n_input = 50                                       #GRU单元输入节点的个数
    n_steps = 1                                        #序列长度
    n_hidden = config.sentence_feature_dim/2           #GRU单元输出节点个数(即隐藏层个数)
    n_classes = 1                                      #类别
    batch_size = 128                                   #小批量大小
    training_step = 5000                               #迭代次数
    display_step  = 200                                #显示步数
    learning_rate = 1e-4                               #学习率 

    model = get_input();

    input_x = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_input])
    input_y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

    hiddens,fw_state,bw_state = single_layer_static_bi_gru(input_x,n_steps,n_hidden)

    #联合训练时无用
    output = tf.contrib.layers.fully_connected(inputs=hiddens[-1],num_outputs=n_classes,activation_fn = tf.nn.softmax)

    #联合训练时由外传进来
    cost = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(output),axis=1))

    #output = random.random();

    #cost = random.random();

    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(output,1),tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

    '''
    test_accuracy_list = []
    test_cost_list=[]
    '''

    with tf.Session() as sess:
        #使用会话执行图
        sess.run(tf.global_variables_initializer())   #初始化变量    
        
        #开始迭代 使用Adam优化的随机梯度下降法
        for i in range(training_step): 
            x_batch = get_batch_x(model, batch_size) 
            y_batch = get_batch_y(batch_size)

            y_batch = y_batch.reshape(batch_size, n_classes)
 
            #Reshape data to get n_steps seq of n_input elements
            x_batch = x_batch.reshape([-1,n_steps,n_input])
            
            #开始训练
            train.run(feed_dict={input_x:x_batch,input_y:y_batch})   
            #train.run(feed_dict={input_x:x_batch})  
            if (i+1) % display_step == 0:
                 #输出训练集准确率  
                training_accuracy,training_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch,input_y:y_batch})       
                #training_accuracy,training_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch})   
                print('Step {0}:Training set accuracy {1},cost {2}.'.format(i+1,training_accuracy,training_cost))

        
        '''
        #全部训练完成做测试  分成200次，一次测试50个样本
        for i in range(200):        
            #x_batch,y_batch = mnist.test.next_batch(batch_size = 50)    
            x_batch = get_batch_x(model, batch_size)  
            y_batch = get_batch_y(batch_size)

            y_batch = y_batch.reshape(batch_size, n_classes)

            #Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1,n_steps,n_input])
            test_accuracy,test_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch,input_y:y_batch})
            #test_accuracy,test_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost) 
            if (i+1)% 20 == 0:
                 print('Step {0}:Test set accuracy {1},cost {2}.'.format(i+1,test_accuracy,test_cost)) 
        print('Test accuracy:',np.mean(test_accuracy_list))
        '''
    
    return hiddens

if __name__ == '__main__':

    get_input()
    #mnist_rnn_classfication()    #1：单层静态双向GRU网络：




    Ruturn:
    A tensor, shape=(batch_size, sentence_feature_dim).
    '''
    pass
