# -*- coding: utf-8 -*-
import tensorflow as tf


class DCNN_seg(object):
    def __init__(self, img_size, num_class, mod, is_training, batch_size):
        self.mod = mod
        self.img_size = img_size
        self.num_class = num_class
        self.is_training = is_training
        self.batch_size = batch_size
        # initialization of weights and bias
        self.weight_init_stddev = 0.1
        self.bias_init_stddev = 0.1
        # hyper-parameters of batch-normalization layer
        self.weight_decay = 0.00004
        self.bn_epslion = 0.0001
        self.bn_decay = 0.9997

    def __w_var(self, name, shape):
        # create weight variables for conv_op or deconv_op
        init = tf.truncated_normal(shape, stddev=self.weight_init_stddev)
        kernel = tf.get_variable(name=name, initializer=init)
        if self.weight_decay is not None:
            tf.add_to_collection('w_loses', tf.multiply(tf.nn.l2_loss(kernel), self.weight_decay, name='w_l'))
        return kernel

    def __b_var(self, name, shape):
        # create bias variables for conv_op or deconv_op
        init = tf.truncated_normal([shape], stddev=self.bias_init_stddev)
        bias = tf.get_variable(name=name, initializer=init)
        # 这里是否需要加l2_loss?
        if self.weight_decay is not None:
            tf.add_to_collection('w_loses', tf.multiply(tf.nn.l2_loss(bias), self.weight_decay, name='w_l'))
        return bias

    def __bn(self, input, name, is_training, activation_fn=None):
        with tf.variable_scope(name, reuse=None):
            return tf.contrib.layers.batch_norm(input, scale=True, activation_fn=activation_fn, is_training=is_training,
                                                trainable=True, updates_collection=None)

    def __conv(self, input, shape, stride=1, atrous=1, bias=False):
        w = self.__w_var(name="conv_w", shape=shape)
        if atrous != 1:
            layer = tf.nn.atrous_conv2d(input, filters=w, rate=atrous, padding='SAME', name='atrous_op')
        else:
            layer = tf.nn.conv2d(input, filter=w, strides=[1, stride, stride, 1], padding='SAME', name='conv_op')
        if bias:
            b = self.__b_var(name='conv_b', shape=shape[-1])
            return tf.nn.bias_add(layer, b)
        else:
            return layer

    def __deconv(self, input, shape, o_size, stride=2):
        w = self.__w_var(name='deconv_w', shape=shape)
        b = self.__b_var(name='deconv_b', shape=shape[-2])
        layer = tf.nn.conv2d_transpose(input, w, o_size, strides=[1, stride, stride, 1], padding='SAME',
                                       name='deconv_op')
        return tf.nn.bias_add(layer, b)

    def __post_processing(self, input):
        prediction = tf.argmax(input, dimension=3, name='prediction')
        # prediction: Type A: int32: [0,c]
        prediction_resize = tf.expand_dims(prediction, dim=3, name='prediction_resize')
        # prediction_resize: Type D: int32: [0,c]
        logits_size = (tf.shape(input)[1], tf.shape(input)[2])
        return input, prediction_resize, logits_size


class U_net(DCNN_seg):
    def __encoder_blk(self, input, inc, ouc):
        with tf.variable_scope('s1', reuse=None):
            net = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='mp')
            net = self.__conv(net, [3, 3, inc, ouc])
            net = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
        with tf.variable_scope('s2', reuse=None):
            net = self.__conv(net, [3, 3, ouc, ouc])
            net = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
        return net

    def __decoder_blk(self, input1, input2, inc, ouc, size):
        crop_input1 = self.__deconv(input1, shape=[2, 2, ouc, inc], o_size=[self.batch_size, size, size, ouc])
        net = tf.concat([crop_input1, input2], axis=3)
        with tf.variable_scope('s1', reuse=None):
            net = self.__conv(net, shape=[3, 3, inc, ouc])
            net = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
        with tf.variable_scope('s2', reuse=None):
            net = self.__conv(net, shape=[3, 3, ouc, ouc])
            net = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
        return net

    def __encoder(self, input):
        with tf.variable_scope('block_0', reuse=None):
            with tf.variable_scope('s1', reuse=None):
                net = self.__conv(input, [3, 3, 3, 16], stride=1)
                net = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
            with tf.variable_scope('s2', reuse=None):
                net = self.__conv(net, [3, 3, 16, 16], stride=1)
                block_0 = self.__bn(net, name='bn', is_training=self.is_training, activation_fn=tf.nn.relu)
            print("after block_0:", block_0.shape)
        with tf.variable_scope('block_1', reuse=None):
            block_1 = self.__encoder_blk(block_0, 16, 32)
            print("after block_1:", block_1.shape)
        with tf.variable_scope('block_2', reuse=None):
            block_2 = self.__encoder_blk(block_1, 32, 64)
            print("after block_2:", block_2.shape)
        with tf.variable_scope('block_3', reuse=None):
            block_3 = self.__encoder_blk(block_2, 64, 128)
            print("after block_3:", block_3.shape)
        with tf.variable_scope('block_4', reuse=None):
            block_4 = self.__encoder_blk(block_3, 128, 256)
            print("after block_4:", block_4.shape)
        return block_0, block_1, block_2, block_3, block_4

    def __decoder(self, b_0, b_1, b_2, b_3, b_4):
        with tf.variable_scope('block_5', reuse=None):
            net = self.__decoder_blk(b_4, b_3, 256, 128, 40)
            print("after block_5:", net.shape)
        with tf.variable_scope('block_6', reuse=None):
            net = self.__decoder_blk(net, b_2, 128, 64, 80)
            print("after block_6:", net.shape)
        with tf.variable_scope('block_7', reuse=None):
            net = self.__decoder_blk(net, b_1, 64, 32, 160)
            print("after block_7:", net.shape)
        with tf.variable_scope('block_8', reuse=None):
            net = self.__decoder_blk(net, b_0, 32, 16, self.img_size)
            print("after block_8:", net.shape)
        return self.__conv(net, [1, 1, 16, self.num_class], stride=1)

    def inference(self, input, reuse):
        print("----------building encoder--------")
        with tf.variable_scope('encoder', reuse=reuse):
            net_0, net_1, net_2, net_3, net_4 = self.__encoder(input)
        print("--------building decoder--------")
        with tf.variable_scope('decoder', reuse=reuse):
            net = self.__decoder(net_0, net_1, net_2, net_3, net_4)
        with tf.variable_scope('post_processing', reuse=reuse):
            a, b, c = self.__post_processing(net)
        return [a], b, c


class DeepLabV3_Resnet50(DCNN_seg):
    pass


class Proposed(DCNN_seg):
    pass


class AdaptSegNet(DCNN_seg):
    pass
