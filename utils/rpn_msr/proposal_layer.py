# -*- coding:utf-8 -*-
import numpy as np
from nms import nms

from utils.bbox.bbox_transform import bbox_transform_inv, clip_boxes
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors
from utils import stat

DEBUG = False

import logging

logger = logging.getLogger("proposal layer")

def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    """
    # 先产生备选框，返回10个不同高度的anchor的4个坐标，是一个 10x4 的数组
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 生成基本的10个anchor

    _num_anchors = _anchors.shape[0]  # 10个anchor，应该是feature map的点数，10

    #im_info: a list of [image_height, image_width, scale_ratios]
    im_info = im_info[0]  # 原始图像的高宽、缩放尺度，0是高度

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'

    pre_nms_topN = cfg.RPN_PRE_NMS_TOP_N  # 12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = cfg.RPN_POST_NMS_TOP_N  # 1000，做完nms之后，最多保留的box的数目
    nms_thresh = cfg.RPN_NMS_THRESH  # nms用参数，阈值是0.7
    min_size = cfg.RPN_MIN_SIZE  # 候选box的最小尺寸，目前是16，高宽均要大于16j,RPN_MIN_SIZE = 8
    # ？？？说是16，为何RPN_MIN_SIZE=8呢？

    logger.debug("传入的前后景概率:%r",rpn_cls_prob_reshape)
    height, width = rpn_cls_prob_reshape.shape[1:3]  # feature-map的高宽 [1:3]=>1,2

    # https://github.com/eragonruan/text-detection-ctpn/issues/311
    # because “rpn_cls_prob_reshape.shape” size is[1,h,w*num_anchor,2]，so
    # width = w*num_anchor/num_anchor num_anchor = 10
    width = width // 10 # ???整除干嘛？//是整除, 10是10个anchor数

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    # rpn_cls_prob_reshape: (1 , H , W , Kx2)，K是10，
    # reshape后，[:, :, :, :, 1]得到的是前景的置信度吧
    # 提取到object的分数，non-object的我们不关心
    scores = np.reshape(
        np.reshape(rpn_cls_prob_reshape,
                   [1, height, width, _num_anchors,2])[:, :, :, :, 1],
                        [1, height, width, _num_anchors])# _num_anchors =10
    logger.debug("去掉了背景(第0列)，只保留前景(第1列)的概率了，shape：%r",scores.shape)

    # 模型输出的pred是相对值，需要进一步处理成真实图像中的坐标
    # 是dx，dy，dw，dh值
    # logger.debug("rpn_bbox_pred网络预测的dx，dy，dw，dh值为：%r",rpn_bbox_pred)
    bbox_deltas = rpn_bbox_pred
    # im_info = bottom[2].data[0, :]

    if DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    if DEBUG:
        print('score map size: {}'.format(scores.shape))

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    # 找个又是那套shift操作，记得不？在train的时候，就操练过，结果是从feature map每个点反向得到原图中的对应的区域的坐标
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors # anchor的数量，10
    K = shifts.shape[0] # 找个是feature map的数量，其实就是HxW
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    # >>>>> 结果维度是[ HxWx10, 4]
    anchors = anchors.reshape((K * A, 4))  # 这里得到的anchor就是整张图像上的所有anchor
    logger.debug('产生最开始的所有的anchor:%r',anchors.shape)

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    # >>>>> 结果维度是[ HxWx10, 4]
    bbox_deltas = bbox_deltas.reshape((-1, 4))  # (HxWxA, 4)
    logger.debug("神经网络预测出来的delta值：%r",bbox_deltas.shape)

    # Same story for the scores:
    # >>>>> 结果维度是[ HxWx10, 1]
    scores = scores.reshape((-1, 1))
    logger.debug("原始前景score的概率分布情况：%s",stat(scores))

    # Convert anchors into proposals via bbox transformations
    # >>>>> anchors     结果维度是[ HxWx10, 4] 4是4个点的坐标
    # >>>>> bbox_deltas 结果维度是[ HxWx10, 4] 4是4个delta值
    # 他们之间一一对应，剩下的额就是要让这个bbox_transform_inv，给还原成对应调整后的框框了
    # 返回的proposals是调整后的2个点的坐标:[x1,y1,x2,y2]
    proposals = bbox_transform_inv(anchors, bbox_deltas)  # 做逆变换，得到box在图像上的真实坐标
    logger.debug("从delta值还原成小框坐标值(2个点,4个值)：%r",proposals.shape)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])  # 将所有的proposal修建一下，超出图像范围的将会被修剪掉
    logger.debug("把超出图像的proposal剪裁掉：%r", proposals.shape)

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size)  # min_size=8,移除那些proposal小于一定尺寸的proposal
    logger.debug("删除掉长和宽小于%d的框，剩余%d个",min_size,len(keep))
    proposals = proposals[keep, :]  # 保留剩下的proposal
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    # ravel展成一维；argsort从低到高排序并返回索引；[::-1]倒过来
    # 看！这里把proposal的顺序打乱了
    # 记住，proposal-bbox_delta-scores是一一对应的，都是对应着feature map的anchor们
    order = scores.ravel().argsort()[::-1]  # score按得分的高低进行排序
    if pre_nms_topN > 0:  # 保留12000个proposal进去做nms
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    bbox_deltas = bbox_deltas[order, :]
    logger.debug("排序前景置信度后，只保留%d个小框了",len(scores))

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    # NMS算法：https://www.cnblogs.com/makefile/p/nms.html
    # 就是一堆候选框，按照概率排序后，跟最大的比较接近的删掉；然后递归再找剩下的
    # 总之就是会删掉一部分框
    keep = nms(np.hstack((proposals, scores)), nms_thresh)  # 进行nms操作，保留2000个proposal
    logger.debug("经过NMS算法，保留的框有%d个",len(keep))
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]
    logger.debug("然后再删除后，保留的框%d个", len(proposals))
    logger.debug("最后剩下的1000个前景score的概率分布情况：%s",stat(scores))

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    blob = np.hstack(
        (scores.astype(np.float32, copy=False),
         proposals.astype(np.float32, copy=False)))

    logger.debug('产生proposal的函数返回:')
    logger.debug('score shape:%r',scores.shape)
    logger.debug('proposals shape:%r', proposals.shape)
    logger.debug('blob shape:%r', blob.shape)

    return blob, bbox_deltas

# min_size=8,移除那些proposal小于一定尺寸的proposal
def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1 # x2-x1
    hs = boxes[:, 3] - boxes[:, 1] + 1 # y2-y1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep # keep是个数组噢~


def _filter_irregular_boxes(boxes, min_ratio=0.2, max_ratio=5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
