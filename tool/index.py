# -*- coding: utf-8 -*-

import tensorflow as tf


def criterion(F1_logits, F2_logits):
    '''
    计算F1_logits和F2_logits之间的差异, 即softmax的差求绝对值之后再求平均值.
    :param F1_logits: Type B : float32 : []
    :param F2_logits: Type B : float32 : []
    :return: int
    '''
    subtract = tf.subtract(tf.nn.softmax(F1_logits), tf.nn.softmax(F2_logits))
    return tf.reduce_mean(tf.abs(subtract))


def accuracy(logits, annotation):
    '''
    计算精度
    :param logits: Type B : float32 : []
    :param annotation: Type A: int32: [0,c]
    :return:
    '''
    annotation_resize_64 = tf.cast(annotation, tf.int64)
    # annotation_resize_64: Type A: int64: [0,c]
    pred_annotation_resize = tf.argmax(logits, dimension=3)
    # pred_annotation_resize = Type A: int64: [0,c]
    correct_pred = tf.equal(annotation_resize_64, pred_annotation_resize)
    # correct_pred: Type A: int64: [0,1]
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def miou(logits, annos, classes):
    anno = tf.cast(annos, tf.int64)
    # anno: Type A: int64: [0,c]
    pred = tf.argmax(logits, dimension=3)
    # pred = Type A: int64: [0,c]
    I = []
    U = []
    pred_1 = tf.equal(pred, 1)
    # pred_i = Type A: BOOL
    anno_1 = tf.equal(anno, 1)
    # anno_i = Type A: BOOL
    pred_0 = tf.equal(pred, 0)
    # anno_i = Type A: BOOL
    anno_0 = tf.equal(anno, 0)
    # anno_i = Type A: BOOL
    cra_iou = tf.reduce_mean(tf.div(tf.reduce_sum(tf.cast(tf.logical_and(pred_1, anno_1), tf.float32)),
                                    tf.reduce_sum(tf.cast(tf.logical_or(pred_1, anno_1), tf.float32))))
    # tf.summary.scalar() 要求输入rank=0, shape=[]的张量, 但是tf.div()输出的是rank=1, shape=[1]的张量
    I.append(tf.reduce_sum(tf.cast(tf.logical_and(pred_1, anno_1), tf.float32)))
    U.append(tf.reduce_sum(tf.cast(tf.logical_or(pred_1, anno_1), tf.float32)))
    I.append(tf.reduce_sum(tf.cast(tf.logical_and(pred_0, anno_0), tf.float32)))
    U.append(tf.reduce_sum(tf.cast(tf.logical_or(pred_0, anno_0), tf.float32)))
    tensor_I = tf.stack(I)
    tensor_U = tf.stack(U)
    return cra_iou, tf.reduce_mean(tf.cast(tf.div(tensor_I, tensor_U), tf.float32))
