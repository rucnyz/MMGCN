# -*- coding: utf-8 -*-
# @Time    : 2022/5/14 12:06
# @Author  : HCY
# @File    : tmp.py
# @Software: PyCharm

import os, sys, tarfile
import numpy as np
from google.colab import drive
import tensorflow.compat.v1 as tf
# drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/Colab_Notebooks/tiktok/")
# with open('Training_dataset/Track2/final_track2_train.txt', "r") as f:
#     final_track2_train = f.read()
# print(final_track2_train[0:1000])
final_track2_train = np.loadtxt('Training_dataset/Track2/final_track2_train.txt')
print(final_track2_train.shape)
# tf.disable_eager_execution()  # 保证sess.run()能够正常运行
# dataset = tf.data.TextLineDataset('Training_dataset/Track2/final_track2_train.txt')
# data_iterator = dataset.make_one_shot_iterator()
# item = data_iterator.get_next()
# with tf.Session() as sess:
#     a = sess.run(item)
#     print(a)


def un_tgz(filename):  # filename是文件的绝对路径
    tar = tarfile.open(filename)
    # 判断是否存在同名文件夹，若不存在则创建同名文件夹：
    if os.path.isdir(os.path.splitext(filename)[0]):
        pass
    else:
        os.mkdir(os.path.splitext(filename)[0])
    tar.extractall(os.path.splitext(filename)[0])
    tar.close()


# tarfile.open('Test_dataset/Track2/final_track2_test_no_answer.txt.tgz')