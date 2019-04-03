# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from bbox import bbox_overlaps
from utils.bbox.bbox_transform import bbox_transform
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors
import tensorflow as tf
import logging

FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger("anchor")
DEBUG = False

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
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride=[16, ], anchor_scales=[16, ],image_name=None):
    logger.debug("开始调用anchor_target_layer，这个函数是来算anchor们和gt的差距")

    logger.debug("传入的参数：")

    logger.debug("rpn_cls_score:%r",type(rpn_cls_score))
    logger.debug("rpn_cls_score:%r", rpn_cls_score.shape)
    logger.debug("gt_boxes:%r", type(gt_boxes))
    logger.debug("gt_boxes:%r", gt_boxes.shape)
    logger.debug("im_info:%r", im_info)
    logger.debug("image_name:%r",image_name.shape)

    """
    ground-truth就是正确的标签的Y的意思，表示的就是正确的标签，错的标签不包含：https://www.zhihu.com/question/22464082
    Assign anchors to ground-truth targets. Produces anchor classification labels
    and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer #这里的A就是文档里面提到的k，就是anchor的个数，F-RNN是9，CTPN是10
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image #这个就是下采样导致的和原图的缩放倍数
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])#缩小16倍么？不是，是anchor本身的大小
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare，是不是包含前景
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives，就是t_x,t_y,t_w,t_h
    #这俩没看懂？？？
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 生成基本的anchor,一共10个，每一个是【x1,y1,y2,y2】坐标形式
    logger.debug("得到了所有的anchor了：%r",_anchors.shape)

    _num_anchors = _anchors.shape[0]  # 10个anchor，shape=[10,4]，4是4个坐标，[x1,y1, x2,y2]
    logger.debug("一共%d个anchors",_num_anchors)

    if DEBUG:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        )))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    # 图片的高度
    im_info = im_info[0]  # 图像的高宽及通道数,[image_height, image_width, scale_ratios]
    if DEBUG:
        print("im_info: ", im_info)
    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # Algorithm:
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W),就是各个anchor包含前景的概率把(1, H, W, Ax2)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽，我怎么觉得是[1:2]啊？我理解错了，1：3，就是index=1和index=2的那两个值
    logger.debug("feature map H/W:(%d,%d)",height,width)

    if DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    # 1. Generate proposals from bbox deltas and shifted anchors
    # 这句话是得到 !!!原图!!! 的对应的每个anchor对应的x
    # width是feature map的宽
    # np.arange生成[0,1,2,...width]的数组，然后*_feat_stride，实际上就是还原到原图中的网格的中心点了，酷！
    shift_x = np.arange(0, width) * _feat_stride #_feat_stride是缩放比例，原图和feature map的，这个相当于是得到原图宽
    logger.debug("shift_x %r",shift_x.shape)

    # 这句话是得到 !!!原图!!! 的对应的每个anchor对应的y
    # height是feature map的高
    shift_y = np.arange(0, height) * _feat_stride
    logger.debug("shift_y %r", shift_y.shape)

    # meshgrid函数用两个坐标轴上的点在平面上画格。[X,Y]=meshgrid(x,y)
    # https://www.cnblogs.com/lemonbit/p/7593898.html
    # meshgrid，就是把x和y分别阔成[|x|,|y|]维度的数组，就是复制多行或者多列
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    logger.debug("变换后shift_x %r", shift_x.shape)
    logger.debug("变换后shift_y %r", shift_y.shape)
    # 这个shift_x是每个行对应的中心点的x坐标，是一个矩阵，对应原图，数量是feature map的shape，每个x，都是对应的中心点x坐标
    # 同理，shift_y是每个列的中心点的x坐标

    # K is H x W ， ravel类似于flatten，就是变成一维的，按照行拼接就成
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), # ravel():将多维数组降位一维
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量
    logger.debug("shift_y,shift_x vstack后 %r", shifts.shape)
    # 得到了啥，是一个矩阵，
    # 4列（因为做了transpose），行是W*H个，但是数间隔是_feat_stride，如下：（不过需要转置）
    # 0, 16, 32....57*16, 0, 16, 32....57*16, .... 0, 16, 32....57*16, 0, 16, 32....57*16，<===一共W*H个
    # 0, 16, 32....37*16, 0, 16, 32....37*16, .... 0, 16, 32....37*16, 0, 16, 32....37*16，<===一共W*H个
    # 0, 16, 32....57*16, 0, 16, 32....57*16, .... 0, 16, 32....57*16, 0, 16, 32....57*16，<===一共W*H个
    # 0, 16, 32....37*16, 0, 16, 32....37*16, .... 0, 16, 32....37*16, 0, 16, 32....37*16，<===一共W*H个
    # 注意，是4列，1，3列是一样的，2，4列是一样的，仔细观察，pls

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 9/10个anchor
    K = shifts.shape[0]  # 50*37，feature-map的宽乘高的大小，对，shitfs矩阵shape现在是[ 50*37, 4 ]

    # _anchors是啥来着？shape[ 10,4 ] ,一共10个，每一个是【x1,y1,y2,y2】坐标形式
    # 看，_anchors和shift加到一起了，我上面就奇怪呢，_anchors是一个0，0开始的相对坐标
    # 到这里，才明白，位移的坐标是在shifts中实现的。
    all_anchors = (
                    _anchors.reshape((1, A, 4)) +
                    shifts.reshape((1, K, 4)).transpose((1, 0, 2)) #K=50*37
                   )  # 相当于复制宽高的维度，然后相加
    logger.debug("得到的all_anchors：%r",all_anchors.shape)
    # transpose((1, 0, 2)，就是转置成为新的数据，第1维和第2维掉个，本来是(0,1,2)=>(1,0,2)):https://blog.csdn.net/AnneQiQi/article/details/60866205
    # 真心晕了，不知道到底矩阵长啥样了？直觉上理解，就是所有的anchor，对应原图的坐标。
    # 今天又看，还是晕，没想清楚，但是一点很清楚：
    # all_anchors肯定是原图对应的那些anchor的坐标了

    all_anchors = all_anchors.reshape((K * A, 4)) # [50*37*10,4]，我理解，就是所有的anchor的坐标
    logger.debug("reshape后的all_anchors：%r", all_anchors.shape)
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 仅保留那些还在图像内部的anchor，超出图像的都删掉
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    logger.debug("图像内的点的索引inds_inside：%r",inds_inside.shape)

    if DEBUG:
        print('total_anchors', total_anchors)
        print('inds_inside', len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor
    logger.debug("图像内部的anchors：%r", anchors.shape)

    if DEBUG:
        print('anchors.shape', anchors.shape)

    # 至此，anchor准备好了
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)
    # labels是一个数据组，长度是所有的在图像内部的anchor的数量，inds_inside是图像内部的anchors
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # 初始化label，均为-1
    logger.debug("labels初始化：%r", labels.shape)

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    # 脑子抽了，明天继续看这块，先跳过去看看简单的。

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    # 计算anchor和gt-box的overlap，用来给anchor上标签
    """
    bbox_overlaps(boxes,query_boxes)
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    这个是对一个gt_box，算9个anchor的重叠度
    """
    # 这里，anchors是所有原图上的feature map为中心点画出的格子对应的10个anchors们
    # 这里，gt_boxes是所有的样本框
    # 这个bbox_overlaps是算GT框和Anchor的相交情况，
    # 返回的内容是每个框和GT的相交比，是一个矩阵 [|anchors|,|gt_box|]，
    # 比如我的anchor是20000，gt是300，那这个数组就是[20000,30]，60万个IoU要计算，我的天哪，怪不得要用C实现呢
    # overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    #     [anchors, gt_boxes] <= [N,K]
    # 行是anchor，列是GT
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),   # anchor是[50*37*10,4]
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    logger.debug("经过bbox_overlaps处理后的overlops:%r",overlaps.shape)
    # 按照他写的注释，overlaps是[50*37*10,|G_t|]的一个数组，大部分都很稀硫啊，这个表示法不好，我理解
    # bbox_overlaps函数居然是用c写的，弄啥嘞，是嫌弃python算这种东西太慢么？

    # print overlap matrix
    # for x in range(0, overlaps.shape[0]):
    #     for y in range(0, overlaps.shape[1]):
    #         if overlaps[x,y]!=0:
    #             logger.debug("x=%d,y=%d:%f", x,y,overlaps[x,y])

    # 存放每一个anchor和每一个gtbox之间的overlap
    # argmax_overlaps的shape[|G|,1]
    argmax_overlaps = overlaps.argmax(axis=1)  # (A)#找到和每一个gtbox，overlap最大的那个anchor
    # for i in range(0, argmax_overlaps.shape[0]):
    #     logger.debug("index=%d", argmax_overlaps[i])

    # ！！！argmax_overlaps是啥？是一个数组，|GT|个，每一个值是这个GT对应的最大的anchor的index
    logger.debug("每行里面，最大的列号argmax_overlaps:%r", argmax_overlaps.shape)
    # 返回是一维数组，长度是行数，值是每列对应的最大的列号
    # 是每行里面，最大的列号

    # inds_inside是在图片范围内的anchor的索引，图像内部的anchor的索引
    # 这里是再做一次图像内部的过滤，去掉那些超范围的anchor的overlap的坐标
    # 注意，这里得到的是坐标
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    logger.debug("每行里面，最大的列号的值max_overlaps:%r", max_overlaps.shape)
    # 这个的坐标是每行里面交面积最大的gt的坐标 ！！！是一个一维数组中
    # [Gt坐标，Gt坐标，Gt坐标，...]，一共anchor数量行
    # sorry,不是坐标，是一个数组，每行和GT最大的交的值

    # 这里是找和某个anchor最交的GT，得到是一个数组，长度是列数，值是行上最大的行号
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # G#找到每个位置上9个anchor中与gtbox，overlap最大的那个
    logger.debug("每列里面，最大的行号gt_argmax_overlaps:%r", gt_argmax_overlaps.shape)
    # [行号,行号,行号,行号...] 一共列数个

    # 先挑行gt_argmax_overlaps，那些行，然后选列
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    logger.debug("每列里面，最大的行号的值gt_max_overlaps:%r", gt_max_overlaps.shape)
    # 得到的是一个一维数组，就是那些最大值

    # np.where得到一个和overlaps一样的true、false矩阵，得到
    # np.where输出满足条件 (即非0) 元素的坐标
    logger.debug("np.where(overlaps == gt_max_overlaps):%r",np.where(overlaps == gt_max_overlaps))
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    logger.debug("np.where(overlaps == gt_max_overlaps)后得到的gt_argmax_overlaps:%r", gt_argmax_overlaps.shape)
    # 过滤完的布尔矩阵，为何只取第一行？??????


    # lables是所有在图像内部的anchors，默认值都是-1，也就是不包含前景
    # fg label: for each gt, anchor with highest overlap
    logger.debug("每个位置上的9个anchor中overlap最大的认为是前景,都打上前景标签1")
    logger.debug("gt_argmax_overlaps:%r",gt_argmax_overlaps)
    labels[gt_argmax_overlaps] = 1   # 每个位置上的9个anchor中overlap最大的认为是前景
    # <------gt_argmax_overlaps是以anchor的某一行的视角来看，找到这个anchor对应的最大的iou的那个GT的index

    # fg label: above threshold IOU，
    logger.debug("overlap大于0.7的认为是前景")
    logger.debug("max_overlaps:%r",max_overlaps)
    labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.7的认为是前景

    # assign bg labels first so that positive labels can clobber them
    # RPN_NEGATIVE_OVERLAP = 0.3
    # 负样本就是IoU小于0.3的，设置值为0
    labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0  # 先给背景上标签，小于0.3overlap的

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # 限制正样本的数量不超过1/2个
    num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)

    # 为何要取第一项[0]：
    # >> > np.where(a == 1)
    # (array([0, 6, 7, 8, 14, 18, 20]),)
    # >> > np.where(a == 1)[0]
    # array([0, 6, 7, 8, 14, 18, 20])
    fg_inds = np.where(labels == 1)[0] #fg_inds，前景的anchors的数量

    logger.debug("保留正样本数：cfg.RPN_FG_FRACTION %f * cfg.RPN_BATCHSIZE %d = %d",cfg.RPN_FG_FRACTION , cfg.RPN_BATCHSIZE,num_fg)
    # logger.debug("fg_inds = %r", fg_inds.shape)
    logger.debug("只保留[%d]个正样本，剩下的正样本去掉，置成-1",num_fg)
    if len(fg_inds) > num_fg:
        # 随机去掉一些样本，去掉就是置为-1，数量是从前景里面去掉要求的数量num_fg，正样本里就只剩下num_fg这么多了
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        labels[disable_inds] = -1  # 变为-1，-1就是disable，就是这次不用的样本，既不能当正样本，也不能当负样本

    #上面是找到正样本，可能找出很多，这个时候要采样

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是256，限制正样本数目最多128，
    # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
    num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]

    # 再从负样本里面去掉一部分，就留 1/2的batch数
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    # 至此， 上好标签，开始计算rpn-box的真值

    # --------------------------------------------------------------
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # anchors是所有的剔除了出界（前面处理过）的剩余的anchors，每条都是anchor的x1,y1,x2,y2
    # argmax_overlaps是指每个GT上，和他overlap最大的那个anchor的索引
    logger.debug("开始计算bbox的差：anchors(图内的)和gt_boxes[argmax_overlaps, :] %r",argmax_overlaps)
    logger.debug("gt_boxes shape=%r",gt_boxes.shape)
    logger.debug("argmax_overlaps=%r",argmax_overlaps)

    # 这里有个细节，argmax_overlaps会有|Anchors|个，也就是大约2万个，
    # 所以，gt_boxes[argmax_overlaps, :]会把gt_boxes撑大了，从原来的300多个，撑成了2万多个
    # 这样，就可以和anchors的数量对上了
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    # 算2万个，我本来还担心速度，后来print了一把，速度很快50毫秒就完成了，就不担心了
    # 返回的是4个差，应该算了一批把？这块感觉是，一口气都算了

    # 定义一个weight的数组，维度是[图内anchor数, 4]，4是x,y,dx,dy
    # 你知道这个inside_weights是干嘛用的么？是为了限定最后4个值，用哪个？如果设成1，就是用，设成0就是不用
    # 你去看model_train.py的loss计算的时候
    # rpn_loss_box_n = tf.reduce_sum(
    #     rpn_bbox_outside_weights * smooth_l1_dist(
    #         rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)),<------------看到这行了吧，也就是x,dx是不参与loss计算的
    #         reduction_indices=[1])
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

    # 那些可以认为是标签的anchor，他的weight设成,[0,1,0,1]，这个之前作者写错了[1,1,1,1]
    # 参考issue：https://github.com/eragonruan/text-detection-ctpn/issues/317
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:')
        print(means)
        print('stdevs:')
        print(stds)

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    # bbox_targets是4个差
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # 外部权重以0填充

    if DEBUG:
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', np.sum(labels == 1))
        print('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    # labels
    labels = labels.reshape((1, height, width, A))  # reshap一下label
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))  # reshape

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    if DEBUG:
        print("anchor target set")

    logger.debug("最后的这个超长的anchor_target_layer返回结果为：")
    logger.debug("rpn_labels:%r",rpn_labels.shape)
    logger.debug("rpn_bbox_targets:%r", rpn_bbox_targets.shape)
    logger.debug("rpn_bbox_inside_weights:%r", rpn_bbox_inside_weights.shape)
    logger.debug("rpn_bbox_outside_weights:%r", rpn_bbox_outside_weights.shape)

    # 我要画出来，GT，和，被选中的anchor
    if FLAGS.debug_mode:
        pass
        from PIL import Image, ImageDraw
        # 先打开原图
        logger.debug("当前处理的文件名字：%s", image_name[0])
        image = Image.open(image_name[0])
        draw = ImageDraw.Draw(image)

        # 先得到anchor是前景的索引
        debug_labels = rpn_labels.reshape(-1,1)
        fg_index  = np.where(debug_labels==1)

        # 拿到每个anchor对应的anchor和gt
        # 其实我不关心所有的GT，我只关心和我IoU>0.7的GT，我把他们画出来
        # 先得到对应的是前景的anchor，这个好获得
        logger.debug("调试画图时候的前景索引：%r",fg_index[0])

        candidate_anchors = anchors[fg_index[0],:]

        # 然后去获得，是前景anchor对应最大IoU的那个GT
        # argmax_overlaps里面就是存着这些gt的index呢，
        # 但是，有重复的，因为他是按照anchor的数量存的，即每个anchor存着他对应最大的gt呢
        # 所以，这么干吧：
        #   1.先通过fg_index过滤所有的argmax_overlaps，只剩下那些选中的anchor
        #   2.然后把这些anchor对应的最大的GT做一个排重操作，剩下的就是我们要的GT们
        left_fg_gt_index_of_anchor = argmax_overlaps[fg_index[0]]
        left_uniq_gt_index = np.unique(left_fg_gt_index_of_anchor)
        candidate_gts = gt_boxes[left_uniq_gt_index,:]

        for anchor in candidate_anchors:
             draw.rectangle(anchor, outline='red')
        for gt in candidate_gts:
             draw.rectangle(gt[:, :4], outline='green')

        import os
        _, fn = os.path.split(str(image_name[0]))
        fn, _ = os.path.splitext(fn)
        if not os.path.exists("data/train/debug"): os.makedirs("data/train/debug")
        dump_img = os.path.join("data/train/debug", fn + '.png')

        image.save(dump_img)

    # 得到一个新的RPN的标签，对比
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

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
    assert gt_rois.shape[1] == 5

    #         (targets_dx, targets_dy, targets_dw, targets_dh)
    # 返回的是4个差:[dx,dy,dw,dh]
    # 其实，dx,dw是没用的啊？！
    # 我理解，算了其实后面loss也不用，恩，等着瞧
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
