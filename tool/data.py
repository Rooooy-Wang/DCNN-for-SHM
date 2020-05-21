# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import argparse


def mcd_trainfile_decoder(file_queue, img_size, batch_size):
    _, serialized_example = tf.TFRecordReader().read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'source_anno': tf.FixedLenFeature([], tf.string),
                                           'source_img': tf.FixedLenFeature([], tf.string),
                                           'target_img': tf.FixedLenFeature([], tf.string)
                                       })
    source_img = tf.reshape(tf.decode_raw(features['source_img'], tf.uint8), [img_size, img_size, 3])
    target_img = tf.reshape(tf.decode_raw(features['target_img'], tf.uint8), [img_size, img_size, 3])
    source_anno = tf.reshape(tf.decode_raw(features['source_anno'], tf.uint8), [img_size, img_size, 1])
    return tf.train.shuffle_batch([source_img, source_anno, target_img],
                                  batch_size=batch_size, num_threads=4,
                                  min_after_dequeue=1,
                                  capacity=1 + 3 * batch_size)


def mcd_evalfile_decoder(file_queue, img_size, batch_size):
    _, serialized_example = tf.TFRecordReader().read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'anno': tf.FixedLenFeature([], tf.string),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'target_img': tf.FixedLenFeature([], tf.string),
                                           'target_anno': tf.FixedLenFeature([], tf.string)
                                       })
    source_img = tf.reshape(tf.decode_raw(features['source_img'], tf.uint8), [img_size, img_size, 3])
    target_img = tf.reshape(tf.decode_raw(features['target_img'], tf.uint8), [img_size, img_size, 3])
    source_anno = tf.reshape(tf.decode_raw(features['source_anno'], tf.uint8), [img_size, img_size, 1])
    target_anno = tf.reshape(tf.decode_raw(features['target_anno'], tf.uint8), [img_size, img_size, 1])
    return tf.train.shuffle_batch([source_img, source_anno, target_img, target_anno], batch_size=batch_size,
                                  num_threads=4, min_after_dequeue=1, capacity=1 + 3 * batch_size)


def file_decoder(file_queue, img_size, batch_size):
    _, serialized_example = tf.TFRecordReader().read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'anno': tf.FixedLenFeature([], tf.string),
                                           'img': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.reshape(tf.decode_raw(features['img'], tf.uint8), [img_size, img_size, 3])
    anno = tf.reshape(tf.decode_raw(features['anno'], tf.uint8), [img_size, img_size, 1])
    return tf.train.shuffle_batch([img, anno], batch_size=batch_size, num_threads=4, min_after_dequeue=1,
                                  capacity=1 + 3 * batch_size)


def get_iopaths(oss):
    if oss:
        parser = argparse.ArgumentParser()
        parser.add_argument('--buckets', type=str, default='', help='input data path')
        parser.add_argument('--checkpointDir', type=str, default='', help='output model path')
        FLAGS, _ = parser.parse_known_args()
        return os.path.join(FLAGS.buckets, 'I'), os.path.join(FLAGS.buckets, 'O')
    else:
        return 'E:/WANGYU/I/', 'E:/WANGYU/O/'


def get_trainpaths(oss, num_tfrecord, train_dataset):
    I, O = get_iopaths(oss)
    model_dir = os.path.join(O, "model/")
    log_dir = os.path.join(O, "logs/")
    trainfile_list = []
    validfile_list = []
    for i in range(num_tfrecord):
        trainfile_list.append(
            os.path.join(I, 'train_data/' + train_dataset + '/train_data' + str(i + 1) + '.tfrecords'))
        validfile_list.append(os.path.join(I, 'train_data/' + train_dataset + '/valid_data.tfrecords'))
    return trainfile_list, validfile_list, model_dir, log_dir


def get_testpaths(oss, num_img, test_dataset, model_loading=None, rate4=None):
    I, O = get_iopaths(oss)
    # model.ckpt.meta 储存了图结构, model.ckpt.data-00000-of-00001, model.ckpt.index 两者储存了图中的变量信息.
    graph_dir = os.path.join(O, 'model/' + model_loading + "/model.ckpt.meta")
    variable_dir = tf.train.latest_checkpoint(os.path.join(O, 'model/' + model_loading))
    testfile_list = []
    testfile_savelist = []
    for i in range(num_img):
        if not rate4:
            testfile_list.append(os.path.join(I, 'test_data/' + test_dataset + '/IMG_' + str(i + 1) + '.JPG'))
            testfile_savelist.append(os.path.join(O, 'test_result/' + model_loading[len(model_loading) - 14:len(
                model_loading)] + '/ANNO_' + str(i + 1) + '.PNG'))
        else:
            for j in range(16):
                testfile_list.append(os.path.join(I, 'test_data/' + test_dataset + '/rate4' + '/IMG_' + str(
                    i + 1) + '_' + str(j) + '.JPG'))
                testfile_savelist.append(os.path.join(O, 'test_result/' + model_loading[len(model_loading) - 8:len(
                    model_loading)] + '/rate4' + '/ANNO_' + str(i + 1) + '_' + str(j + 1) + '.PNG'))
    return testfile_list, testfile_savelist, graph_dir, variable_dir


def get_train_data(mod, trainfile_list, img_size, batch_size):
    with tf.name_scope('train_data'):
        trainfile_que = tf.train.string_input_producer(trainfile_list, num_epochs=None)
        if mod == 'train':
            return file_decoder(trainfile_que, img_size, batch_size)
        elif mod == 'MCD_train':
            return mcd_trainfile_decoder(trainfile_que, img_size, batch_size)
        else:
            return False


def get_valid_data(mod, evalfile_list, img_size, batch_size):
    with tf.name_scope('eval_data'):
        evalfile_que = tf.train.string_input_producer(evalfile_list, num_epochs=None)
        if mod == 'train':
            return file_decoder(evalfile_que, img_size, batch_size)
        elif mod == 'MCD_train':
            return mcd_evalfile_decoder(evalfile_que, img_size, batch_size)
        else:
            return False
