"this network was designed by Wang"
"Author:MingBo Hong"

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

l2 = 0.0001
decay_step = 10000               # 衰减迭代数
learning_rate_decay_factor = 0.1  # 学习率衰减因子
initial_learning_rate = 0.1      # 初始学习率
moving_average_decay = 0.9999
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)
def avgPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """avg-pooling"""
    return tf.nn.avg_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)
def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)


def batch_norm_layer(value, is_training=False, name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果

    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        # 训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, updates_collections=None, is_training=True)
    else:
        # 测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, updates_collections=None, is_training=False)

def fcLayer(x, outputD, reluFlag, name,l2):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        output_size = x.get_shape().as_list()
        w = tf.get_variable("w", shape = [output_size[1], outputD],initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01))
        b = tf.get_variable("b", [outputD], initializer=tf.constant_initializer(0.0))
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if l2 is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), l2, name='weight_loss')
            tf.add_to_collection(name='losses', value=weight_decay)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum],initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.01))
        b = tf.get_variable("b", shape = [featureNum],initializer=tf.constant_initializer(0.0))

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

class alexNet(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum,is_training):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.buildCNN(is_training)


    def buildCNN(self,is_training):
        """build model"""
        conv1 = convLayer(self.X, 5, 5, 1, 1, 48, "conv1", "SAME")
        pool1 = avgPoolLayer(conv1, 3, 3, 2, 2, "pool1", "SAME")
        bn1 = batch_norm_layer(pool1,is_training,'BN1')
        #lrn1 = LRN(pool1, 3, 2e-05, 0.75, "norm1")


        conv2= convLayer(bn1, 5, 5, 1, 1, 48, "conv2","SAME")
        pool2 = avgPoolLayer(conv2, 3, 3, 2, 2, "pool2", "SAME")
        bn2 = batch_norm_layer(pool2, is_training,'BN2')


        conv3 = convLayer(bn2, 5, 5, 1, 1, 96, "conv3","SAME")
        pool3 = maxPoolLayer(conv3, 3, 3, 2, 2, "pool3", "SAME")
        bn3 = batch_norm_layer(pool3, is_training,'BN3')




        conv4 = convLayer(bn3, 5, 5, 1, 1, 96, "conv4","SAME")
        pool4 = maxPoolLayer(conv4, 3, 3, 2, 2, "pool4", "SAME")

        output_size = pool4.get_shape().as_list()

        nodes = output_size[1] * output_size[2] * output_size[3]
        reshaped = tf.reshape(pool4, [-1, nodes])



        fc_1 = fcLayer(reshaped, 256, True, "fc5",l2)
        if is_training:
            dropout1 = dropout(fc_1, self.KEEPPRO)
            self.fc_2 = fcLayer(dropout1, self.CLASSNUM, True, "fc6",l2)
        else:
            self.fc_2 = fcLayer(fc_1, self.CLASSNUM, True, "fc6",l2)


def losses_summary(total_loss):
    average_op = tf.train.ExponentialMovingAverage(decay=0.9) #创建一个新的指数滑动均值对象
    losses = tf.get_collection(key='losses')# 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失
    # 创建‘shadow variables’,并添加维护滑动均值的操作
    maintain_averages_op = average_op.apply(losses+[total_loss])#维护变量的滑动均值，返回一个能够更新shadow variables的操作
    for i in losses+[total_loss]:
        tf.summary.scalar(i.op.name+"_raw", i) #保存变量到Summary缓存对象，以便写入到文件中
        tf.summary.scalar(i.op.name, average_op.average(i)) #average() returns the shadow variable for a given variable.
    return maintain_averages_op  #返回损失变量的更新操作

def loss(logits, labels):
    labels = tf.cast(x=labels, dtype=tf.int32)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
    tf.add_to_collection(name='losses', value=cross_entropy_loss)
    return tf.add_n(inputs=tf.get_collection(key='losses'), name='total_loss')


def one_step_train(total_loss, step):
    lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                    global_step=step,
                                    decay_steps=decay_step,
                                    decay_rate=learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar("learning_rate", lr)
    losses_movingaverage_op = losses_summary(total_loss)
    #tf.control_dependencies是一个context manager,控制节点执行顺序，先执行control_inputs中的操作，再执行context中的操作
    with tf.control_dependencies(control_inputs=[losses_movingaverage_op]):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        #trainer = tf.train.AdamOptimizer(learning_rate=lr)
        gradient_pairs = trainer.compute_gradients(loss=total_loss) #返回计算出的（gradient, variable） pairs
    gradient_update = trainer.apply_gradients(grads_and_vars=gradient_pairs, global_step=step) #返回一步梯度更新操作

    variables_average_op = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=step)
    # tf.trainable_variables() 方法返回所有`trainable=True`的变量，列表结构
    maintain_variable_average_op = variables_average_op.apply(var_list=tf.trainable_variables())#返回模型参数变量的滑动更新操作
    with tf.control_dependencies(control_inputs=[gradient_update, maintain_variable_average_op]):
        gradient_update_optimizor = tf.no_op() #Does nothing. Only useful as a placeholder for control edges
    return gradient_update_optimizor