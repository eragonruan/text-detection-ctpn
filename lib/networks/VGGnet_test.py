import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
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

        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        (self.feed('rpn_conv/3x3').lstm(512, 128, name='lstm_o'))
        (self.feed('lstm_o').lstm_bbox(128, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_bbox(128, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))
        #(self.feed('lstm_o').fc_bbox(256, name='fc_box'))
        #(self.feed('fc_box').fc_bbox(len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        #(self.feed('fc_box').fc_bbox(len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        '''
        (self.feed('conv5_3')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 10 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_conv/3x3')
         .conv_rpn(1, 1, len(anchor_scales) * 10 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))
        '''
        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))


        '''
        (self.feed('conv5_3', 'rois')
         .roi_pool(7, 7, 1.0 / 16, name='pool_5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

        (self.feed('fc7')
         .fc(n_classes * 4, relu=False, name='bbox_pred'))
        '''
