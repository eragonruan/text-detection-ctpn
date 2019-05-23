# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from bbox import bbox_overlaps
from utils.bbox.bbox_transform import bbox_transform
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors
import tensorflow as tf
import logging
from utils import stat

FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger("anchor")
DEBUG = False

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride=[16, ], anchor_scales=[16, ],image_name=None):
    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'
    height, width = rpn_cls_score.shape[1:3]

    return anchor_target_layer_process((width,height), gt_boxes, im_info, _feat_stride, anchor_scales,image_name,FLAGS.debug)

# bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
# cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
# cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化
# _anchor_target_layer 主要功能是计算获得属于rpn网络的label。
# https://zhuanlan.zhihu.com/p/32230004
# 通过对所有的anchor与所有的GT计算IOU，
# 由此得到 rpn_labels, rpn_bbox_targets,
# rpn_bbox_inside_weights, rpn_bbox_outside_weights
# 这4个比较重要的第一次目标label，通过消除在图像外部的 anchor，
# 计算IOU >=0.7 为正样本，IOU <0.3为负样本，
# 得到在理想情况下应该各自一半的256个正负样本
# （实际上正样本大多只有10-100个之间，相对负样本偏少）。
# rpn_cls_score是啥，是神经网络跑出来的一个分类结果，是包含文字，还是不包含文字的一个概率值，
#       因为有9个框，而且有包含和不包含2个值，所以是(1, H, W, Ax2)维度的，对H,W的含义是，对每一个feature map中的点，都做了预测
# 另，这个太神奇了，参数本来都是张量
def anchor_target_layer_process(feature_map_shape, gt_boxes, im_info, _feat_stride, anchor_scales,image_name,is_debug):
    logger.debug("feature map shape:%r",feature_map_shape)
    logger.debug("gt_boxes:%r", type(gt_boxes))
    logger.debug("gt_boxes:%r", gt_boxes.shape)
    logger.debug("im_info:%r", im_info)
    logger.debug("image_name:%r",image_name.shape)

    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 生成基本的anchor,一共10个，每一个是【x1,y1,y2,y2】坐标形式
    _num_anchors = _anchors.shape[0]  # 10个anchor，shape=[10,4]，4是4个坐标，[x1,y1, x2,y2]
    _allowed_border = 0
    im_info = im_info[0]  # 图像的高宽及通道数,[image_height, image_width, scale_ratios]
    width,height = feature_map_shape  # feature-map的高宽，我怎么觉得是[1:2]啊？我理解错了，1：3，就是index=1和index=2的那两个值

    # 产生所有的anchors，并剔除超出图像范围的
    shift_x = np.arange(0, width) * _feat_stride #_feat_stride是缩放比例，原图和feature map的，这个相当于是得到原图宽
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), # ravel():将多维数组降位一维
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量
    A = _num_anchors  # 9/10个anchor
    K = shifts.shape[0]  # 50*37，feature-map的宽乘高的大小，对，shitfs矩阵shape现在是[ 50*37, 4 ]
    all_anchors = (
                    _anchors.reshape((1, A, 4)) +
                    shifts.reshape((1, K, 4)).transpose((1, 0, 2)) #K=50*37
                   )  # 相当于复制宽高的维度，然后相加
    logger.debug("所有anchors：%r",all_anchors.shape)
    all_anchors = all_anchors.reshape((K * A, 4)) # [50*37*10,4]，我理解，就是所有的anchor的坐标
    total_anchors_num = int(K * A)
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor
    logger.debug("图像内部anchors：%r", anchors.shape)

    #label: 1正例,0负例,-1不相关
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # 初始化label，均为-1
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),   # anchor是[50*37*10,4]
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    logger.debug("计算了IoU，anchors：%r,gt:%r",anchors.shape,gt_boxes.shape)
    argmax_overlaps = overlaps.argmax(axis=1)  # (A)#找到和每一个gtbox，overlap最大的那个anchor
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # G#找到每个位置上9个anchor中与gtbox，overlap最大的那个
    labels[gt_argmax_overlaps] = 1   # 每个位置上的9个anchor中overlap最大的认为是前景
    __debug_iou_max_with_gt_anchors = anchors[gt_argmax_overlaps]
    labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.7的认为是前景
    __debug_iou_more_0_7_anchors = anchors[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP]
    logger.debug("经过IoU>0.7筛选，现在有%d个前景样本（anchors）",(labels==1).sum())
    labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0  # 先给背景上标签，小于0.3overlap的
    logger.debug("overlap小于0.3的认为是背景，一共有%d个", (max_overlaps < cfg.RPN_NEGATIVE_OVERLAP).sum())

    # 开始按照batch要求，削减样本正样本数量，以及增加负样本
    num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0] #fg_inds，前景的anchors的数量
    if len(fg_inds) > num_fg:
        # 随机去掉一些样本，去掉就是置为-1，数量是从前景里面去掉要求的数量num_fg，正样本里就只剩下num_fg这么多了
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        labels[disable_inds] = -1  # 变为-1，-1就是disable，就是这次不用的样本，既不能当正样本，也不能当负样本
        logger.debug("经过最大正样本数量600限定，现有%d个前景样本（anchors）", (labels == 1).sum())


    num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        logger.debug("经过设置负样本后，现在有%d个前景样本（anchors）", (labels == 1).sum())

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        positive_weights = (cfg.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights

    inside_labels = labels.copy()
    labels = _unmap(labels, total_anchors_num, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, total_anchors_num, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors_num, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors_num, inds_inside, fill=0)  # 外部权重以0填充
    labels = labels.reshape((1, height, width, A))  # reshap一下label
    rpn_labels = labels
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))  # reshape
    rpn_bbox_targets = bbox_targets
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    logger.debug("rpn_labels:%r",rpn_labels.shape)
    logger.debug("rpn_labels中正例%d个,负例%d个,无效%d个", (rpn_labels==1).sum(),(rpn_labels==0).sum(),(rpn_labels==-1).sum())
    logger.debug("rpn_bbox_targets:%r", rpn_bbox_targets.shape)
    logger.debug("rpn_bbox_inside_weights:%r", rpn_bbox_inside_weights.shape)
    logger.debug("rpn_bbox_outside_weights:%r", rpn_bbox_outside_weights.shape)

    if is_debug: debug_draw(__debug_iou_max_with_gt_anchors, __debug_iou_more_0_7_anchors, anchors, gt_boxes, image_name,
               inside_labels)

    # 得到一个新的RPN的标签，对比
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

#  调试用，把所有的anchor都画出来看看，还有GT
def debug_draw(__debug_iou_max_with_gt_anchors,
               __debug_iou_more_0_7_anchors,
               anchors,
               gt_boxes,
               image_name,
               inside_labels):

    # 这段代码是调试用的，是为了让训练过程可视化，
    # 我要画出来，所有的备选anchor，包括IoU>0.7的 + GT相交最多的那个anchor，这些都是样本备选，用红色
    # 我要画出来所有的GT，用绿色
    # 我还要画出最后的选中的备选的正例和负例，用蓝色，用黑色（？蓝色可能看不到吧，因为被红色重复了，那我都他们都错位1个像素），

    logger.debug("调试画图时候的前景索引")
    from PIL import Image, ImageDraw
    # 先打开原图
    logger.debug("当前处理的文件名字：%s", image_name[0])
    image = Image.open(image_name[0])
    draw = ImageDraw.Draw(image)

    # 根据索引得到，要画的前景anchor，inside_labels是因为label现在已经包含了超出图片范围的anchor的指示了
    positive_anchors = anchors[inside_labels == 1, :]
    negative_anchors = anchors[inside_labels == 0, :]

    RED = "#FF0000"
    BLUE= "#0000FF"
    PURPLE = "#9900FF"
    GREEN = "#00FF00"
    GRAY = "#808080"

    # draw.rectangle(__debug_specail_anchors.tolist(), outline=RED)
    # logger.debug("__debug_specail_gt:%r",__debug_specail_gt)
    # draw.rectangle(__debug_specail_gt[:4], outline=RED)

    # logger.debug("[调试画图] 画出所有IoU大于0.7的anchors[%d]，红色",len(__debug_iou_more_0_7_anchors))
    # for anchor in __debug_iou_more_0_7_anchors:
    #     draw.rectangle(anchor.tolist(), outline=RED)
    #
    # logger.debug("[调试画图] 画出所有和GT相交最大的anchor[%d]，紫色",len(__debug_iou_max_with_gt_anchors))
    # for anchor in __debug_iou_max_with_gt_anchors:
    #     draw.rectangle(anchor.tolist(), outline=PURPLE)

    logger.debug("[调试画图] 画出所有的GT[%d]，绿色",len(gt_boxes))
    for gt in gt_boxes:
        draw.rectangle(gt[:4].tolist(), outline=GREEN)


    logger.debug("[调试画图] 画出所有正例[%d]，蓝色",len(positive_anchors))
    for anchor in positive_anchors:
        draw.rectangle(anchor.tolist(), outline=BLUE)

    # logger.debug("[调试画图] 画出所有负例[%d]，灰色",len(negative_anchors))
    # for anchor in negative_anchors:
    #     draw.rectangle(anchor.tolist(), outline=GRAY)


    # 保存图片
    import os
    _, fn = os.path.split(str(image_name[0]))
    fn, _ = os.path.splitext(fn)
    if not os.path.exists("data/train/debug"): os.makedirs("data/train/debug")
    dump_img = os.path.join("data/train/debug", fn + '.png')
    image.save(dump_img)


# 这个函数说白了，就是还原到大矩阵，把那些抛弃的都拿回来
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # 如果是1维度的
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

# ex_rois就是anchor的4个坐标
# gt_rois就是gt的4个坐标
# 那你算啥呢？
# 你算dx,dy,dw,dh，这个4元组就是标签啊
# 不过，dx,dw是在CTPN的算法是没用的啊？这点我表示困惑
# 2019-04-03 16:10:08,510 : DEBUG : ex_rois:anchors:(20008, 4)
# 2019-04-03 16:10:08,510 : DEBUG : gt_rois:gts:(20008, 5)
# 看！两个的行数是一样的，啥意思？就是gt多少个，anchor就给他准备多少个，
# 你说了，怎么20008，2万多个，怎么这么多？我给你算算：
# 870x662的图，featuremap是54x41=2214，x10个anchor是22140，再取出越界的，剩下这20008个了。
# 我去，都比啊，恩！
# 哦，我理解，错了，这个不是在算IoU，而是在算那4个值,x,y,dw,dh，
# 没关系，注释不删了，有助于理解别的
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    logger.debug("_compute_targets")
    logger.debug("ex_rois:anchors:%r",ex_rois.shape)
    logger.debug("gt_rois:gts:%r", gt_rois.shape)

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    # assert gt_rois.shape[1] == 5,第五列是 可能性，其实这个校验没必要，我在写image_debug的时候去掉了这点，导致报错，所以注释掉他

    #         (targets_dx, targets_dy, targets_dw, targets_dh)
    # 返回的是4个差:[dx,dy,dw,dh]
    # 其实，dx,dw是没用的啊？！
    # 我理解，算了其实后面loss也不用，恩，等着瞧
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
