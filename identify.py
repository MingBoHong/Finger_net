import tensorflow as tf
import train
from scipy import misc
import math
import alexnet
import preprocess
import numpy as np
import preprocess
C_size = 32
T = 0
F = 0
checkpoint_path = '/model/checkpoint'
def cut_image(image):
    data =[]
    var = 0
    number_slice = (image.shape[0]* image.shape[1])/(C_size*C_size)
    for row in range(int(image.shape[0] / C_size)):
        for col in range(int((image.shape[1]) / C_size)):
            row_start = row * C_size
            row_end = (row+1) * C_size
            col_start = col * C_size
            col_end = (col+1) * C_size
            var = var + np.var(image[row_start:row_end, col_start:col_end])

    for row in range(int(image.shape[0] /C_size)):
        for col in range(int((image.shape[1]) /C_size)):
            row_start = row * C_size
            row_end = (row + 1) * C_size
            col_start = col * C_size
            col_end = (col + 1) * C_size
            if np.var(image[row_start:row_end, col_start:col_end]) > (var / number_slice) * 0.5:
                data.append(image[row_start:row_end, col_start:col_end])

    return np.array(data)
def pad_image(ori_image):

    if math.ceil(ori_image.shape[0] / C_size) > int(ori_image.shape[0] / C_size):
        pad_height = math.ceil(ori_image.shape[0] / C_size)*32 - ori_image.shape[0]
    else:
        pad_height = 0
    if math.ceil(ori_image.shape[1]/C_size) > int(ori_image.shape[1]/C_size):
        pad_width = math.ceil(ori_image.shape[1] / C_size)*32 - ori_image.shape[1]
    else:
        pad_width = 0
    return np.pad(ori_image, ((math.ceil(pad_height/2), int(pad_height/2)), (math.ceil(pad_width/2), int(pad_width/2))), 'constant', constant_values=255)



def identify(image):
    with tf.Graph().as_default():
        pic = pad_image(image)
        data = cut_image(pic)
        image = tf.convert_to_tensor(data)
        image = tf.expand_dims(image,-1)
        image = tf.cast(image, dtype=tf.float32)


        Finger_model = alexnet.alexNet(image, 0.5, 2, False)
        logits = Finger_model.fc_2
        moving_average_op = tf.train.ExponentialMovingAverage(decay=alexnet.moving_average_decay)
        variables_to_restore = moving_average_op.variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)
        with tf.Session() as sess:
            checkpoint_proto = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path)
            if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
                saver.restore(sess, checkpoint_proto.model_checkpoint_path)
            else:
                print('checkpoint file not found!')
                return
            predict = sess.run(logits)
            predict = np.array(predict)

            """
             Vote Mechanism
            """

            Num_0 = sum(predict == 0)
            Num_1 = sum(predict == 1)
            if Num_0[0] <= Num_1[0]:
                return "Fake"
            else:
                return "Alive"






