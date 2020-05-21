# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from network import *
import tool as tl
import datetime
import os

MOD = 'train'
# 训练步数
MAX_ITERATION = 20000
# 每隔多少步记录训练结果
SUM_OP_ITERATION = 2000
# 每隔多少步存储模型
SAVE_OP_ITERATION = 5000
# 初始学习率
LR = 0.001
# 学习率下降频率(参考MCD)
DECAY_STEP = MAX_ITERATION // 10
# 学习率下降程度(参考MCD)
DECAY_RATE = 0.85
# 小批量数
BATCH_SIZE = 16
# 类别数
NUM_CLASS = 2
# 图像尺寸
IMG_SIZE = 320
# 是否在OSS平台训练
OSS = False
# 创建时间戳
TIME_STAMP = str(datetime.datetime.now()).replace(' ', '-').replace('.', '-').replace(':', '-')


def train(train_dataset, model):
    with tf.name_scope("data"):
        trainfile_list, validfile_list, model_dir, log_dir = tl.get_trainpaths(OSS, 7, train_dataset)
        train_imgs, train_annos = tl.get_train_data('train', trainfile_list, IMG_SIZE, BATCH_SIZE)
        valid_imgs, valid_annos = tl.get_valid_data('train', validfile_list, IMG_SIZE, BATCH_SIZE)
    with tf.name_scope("input"):
        img = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3], name='img')
        anno = tf.placeholder(tf.int32, shape=[None, IMG_SIZE, IMG_SIZE, 1], name='anno')
        is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.placeholder(tf.int64, name='global_step')
    with tf.name_scope("output"):
        # 初始化神经网络
        net = U_net(IMG_SIZE, NUM_CLASS, 'train', is_training, BATCH_SIZE)
        # 获得神经网络输出
        logits_list, prediction, outputsize = net.inference(img, False)
        # src_prediction: Type A: int32: [0,c]
        logits = logits_list[0]
    with tf.name_scope("summary_op"):
        print("--------setting up summary_op--------")
        anno_resize = tf.squeeze(anno, squeeze_dims=[3])
        # src_anno_resize: Type A: int32: [0,c]
        # 计算精度(accuracy)并进行summary
        acc = tl.accuracy(logits, anno_resize)
        tf.add_to_collection('summary_op', tf.summary.scalar("accuracy", acc))
        # 计算crack对象的iou值并进行summary
        craiou, miou = tl.miou(logits, anno_resize, NUM_CLASS)
        tf.add_to_collection('summary_op', tf.summary.scalar('cra_iou', craiou))
        tf.add_to_collection('summary_op', tf.summary.scalar("miou", miou))
        summary_op = tf.summary.merge_all()
    with tf.name_scope("loss_function"):
        cre = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=anno_resize, name='cre'))
        # tf.nn.sparse_softmax_cross_entropy_with_logits(Type B, Type A)
        tf.add_to_collection('a_summary_op', tf.summary.scalar('cross_entropy', cre))
        # 计算总的损失函数
        l2_reg = tf.add_n(tf.get_collection('w_loses'))
        loss = l2_reg + cre
    with tf.name_scope("train_op"):
        print("--------setting up train_op--------")
        lr = tf.train.exponential_decay(learning_rate=LR, global_step=global_step, decay_steps=DECAY_STEP,
                                        decay_rate=DECAY_RATE, staircase=True, name='lr')
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
        train_op = optimizer.apply_gradients(grads)
    with tf.name_scope("writer"):
        print("Setting up Writer...")
        # 当tf.train.Saver()的var_list为None时, 将保存所有可以保存的变量, 因此bn层中的moving means 和moving variances 也可以?
        # 初始化train_writer 用于summary 训练结果, 以及记录部分阶段的训练过程(checkpoint)
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, TIME_STAMP + '/train'), tf.Session().graph)
        valid_writer = tf.summary.FileWriter(os.path.join(log_dir, TIME_STAMP + '/valid'))
    with tf.name_scope("saver"):
        print("--------setting up saver--------")
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True))
    with tf.Session(config=config) as sess:
        print("--------start training--------")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 多线程的初始化
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                save_num = 0
                for itr in xrange(MAX_ITERATION):
                    # 获取训练数据
                    train_img_bh, valid_anno_bh = sess.run([train_imgs, train_annos])
                    train_fd = {img: train_img_bh, anno: valid_anno_bh, is_training: False, global_step: itr}
                    # 执行训练和summary操作
                    if itr % SAVE_OP_ITERATION == 0:
                        _ = sess.run([train_op], feed_dict=train_fd)
                        if not OSS:
                            os.makedirs(os.path.join(model_dir, TIME_STAMP + '/' + str(save_num)))
                        saver.save(sess, os.path.join(model_dir, TIME_STAMP + '/' + str(save_num) + '/model.ckpt'))
                        save_num = save_num + 1
                    elif itr % SUM_OP_ITERATION == 0:
                        # 训练该数据集并summary
                        summary_str, _ = sess.run([summary_op, train_op], feed_dict=train_fd)
                        train_writer.add_summary(summary_str, itr)
                        # 在源域上验证网络并summary
                        valid_img_bh, valid_anno_bh = sess.run([valid_imgs, valid_annos])
                        v_fd = {img: valid_img_bh, anno: valid_anno_bh, is_training: False}
                        summary_str = sess.run(summary_op, feed_dict=v_fd)
                        valid_writer.add_summary(summary_str, itr)
                    else:
                        _ = sess.run([train_op], feed_dict=train_fd)
                # 终止writer进程
                train_writer.close()
                valid_writer.close()
                # 保存模型
                if not OSS:
                    os.makedirs(os.path.join(model_dir, TIME_STAMP + '/' + str(save_num)))
                saver.save(sess, os.path.join(model_dir, TIME_STAMP + '/' + str(save_num) + '/model.ckpt'))
                break
        except tf.errors.OutOfRangeError:
            print('Done train after all data')
        finally:
            coord.request_stop()
            coord.join(threads)


def mcd_train(train_dataset, model):
    pass


def test(test_dataset, model_loading):
    pass


def mcd_test(test_dataset, model_loading):
    pass


def main():
    if MOD == 'train':
        train(train_dataset='TSR_train', model='U_Net')
    elif MOD == 'MCD_train':
        mcd_train(train_dataset='TSR_train', model='U_Net')
    elif MOD == 'test':
        test(test_dataset='TSR_test', model_loading='')
    else:
        mcd_test(test_dataset='TSR_test', model_loading='')


if __name__ == "__main__":
    tf.app.run()
