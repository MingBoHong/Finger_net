import shutil
import tensorflow as tf
from scipy import misc
import os
import numpy as np
import math
import threading
import random
C_size = 32

Binary = ['img_false','img_true']
Binary_dir = ['Fake','True']

lockA = threading.Lock()
lockB = threading.Lock()
flag_A = True
flag_B = True



def transform_data_A(image_path):

    global flag_A
    try:
        sample_path = os.path.join(image_path, Binary_dir[0])
        for lists in os.listdir(sample_path):
            lockA.acquire()
            sub_path = os.path.join(sample_path, lists)
            pic = misc.imread(sub_path)
            img_raw = pic.tobytes()  # 将图片转化为原生byte
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串
            if (lists==os.listdir(sample_path)[-1]):
                flag_A = False
            if (flag_B):
                lockB.release()
            else:
                lockA.release()

    except FileNotFoundError:
        print("File is not found.")




def transform_data_B(image_path):
    global flag_B
    try:
        sample_path = os.path.join(image_path, Binary_dir[1])
        for lists in os.listdir(sample_path):
            lockB.acquire()
            sub_path = os.path.join(sample_path, lists)
            pic = misc.imread(sub_path)
            img_raw = pic.tobytes()  # 将图片转化为原生byte
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串
            if (lists==os.listdir(sample_path)[-1]):
                 flag_B = False
            if(flag_A):
                lockA.release()
            else:
                lockB.release()
    except FileNotFoundError:
        print("File is not found.")

def shuffle_v1():
    """
    双线程处理数据，线程同步互锁，速度贼鸡儿慢 = =
    """
    writer = tf.python_io.TFRecordWriter(r"F:\KDR\BRL\project\Train_image/train.tfrecords")
    lockB.acquire()
    t1 = threading.Thread(target=transform_data_A, args=(r"F:\KDR\BRL\project\Train_image",'train.tfrecords'))
    t2 = threading.Thread(target=transform_data_B, args=(r"F:\KDR\BRL\project\Train_image",'train.tfrecords'))
    t1.start()
    t2.start()
    t1.join()
    t2.join()








