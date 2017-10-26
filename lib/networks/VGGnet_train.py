# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from .network import Network
from ..fast_rcnn.config import cfg

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):

        # n_classes = 21
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        # zai 5-3 te zheng ceng yong yi ge 3*3 de huan dong chuan kou
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))


        (self.feed('rpn_conv/3x3').lstm(512,128,name='lstm_o'))
        (self.feed('lstm_o').lstm_bbox(128,len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_bbox(128,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))
        #(self.feed('lstm_o').fc_bbox(256, name='fc_box'))
        #(self.feed('fc_box').fc_bbox(len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        #(self.feed('fc_box').fc_bbox(len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        # Loss of rpn_cls & rpn_boxes
        # shape is (1, H, W, A x 4) and (1, H, W, A x 2)
        # 加入全卷积层，用来预测anchor的相对位置，也即delta
        '''
        (self.feed('rpn_conv/3x3')
             .conv_rpn(1,1,len(anchor_scales) * 10 * 4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))
        # 加入全卷积层，用来预测每一个delta的得分，object和non-object两个得分
        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales) * 10 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))
        '''
        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        #把得分reshape回正常的shape
        (self.feed('rpn_cls_prob')
             .spatial_reshape_layer(len(anchor_scales)*10*2, name = 'rpn_cls_prob_reshape'))

        # 生成固定anchor，并给所有的anchor加上之前得到的rpn-bbox-pred，也就是delta
        # 在做nms之类的处理，最后得到2000个rpn-rois
        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name = 'rpn_rois_data'))

        # matching boxes and groundtruth,
        # and randomly sample some rois and labels for RCNN
        # 在之前生成的2000个proposal中挑选一部分，并上标签，准备送入rcnn
        (self.feed('rpn_rois','rpn_targets','gt_boxes', 'gt_ishard', 'dontcare_areas')
             .proposal_target_layer(n_classes,name = 'roi-data'))
