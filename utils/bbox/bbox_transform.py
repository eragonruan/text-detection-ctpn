import numpy as np
import logging

logger = logging.getLogger("bbox transform")

def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'.format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    logger.debug("计算完的bbox regression结果：%r",targets.shape)
    return targets

# >>>>> anchors     结果维度是[ HxWx10, 4] 4是4个点的坐标
# >>>>> bbox_deltas 结果维度是[ HxWx10, 4] 4是4个delta值
# 他们之间一一对应，剩下的额就是要让这个bbox_transform_inv，给还原成对应调整后的框框了
def bbox_transform_inv(boxes, deltas):
    # debug完成后，要删掉这个，太TMD多了
    # 我主要想看看dw,dx到底是啥
    # logger.debug("bbox_transform: boxes:%r" ,boxes)
    # logger.debug("bbox_transform: deltas:%r", deltas)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths # anchor的中心位置
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # 我靠，我一直想知道，为何要预测4个，不是CTPN算法只需要预测d_h和d_y么？d_x,d_w是不需要的
    # 至此，我终于明白了，你预测吧，我根本就不用！！！纳尼？！！！
    # 那会不会对权重有影响呢？！我有点想不清楚，忽略dx,dw，会影响梯度下降算法吗？我不知道。。。
    # 别担心：看下面的解答：
    # 你知道这个inside_weights是干嘛用的么？是为了限定最后4个值，用哪个？如果设成1，就是用，设成0就是不用
    # 你去看model_train.py的loss计算的时候
    # rpn_loss_box_n = tf.reduce_sum(
    #     rpn_bbox_outside_weights * smooth_l1_dist(
    #         rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)),<------------看到这行了吧，也就是x,dx是不参与loss计算的
    #         reduction_indices=[1])
    # 那些可以认为是标签的anchor，他的weight设成,[0,1,0,1]，这个之前作者写错了[1,1,1,1]
    # 参考issue：https://github.com/eragonruan/text-detection-ctpn/issues/317
    pred_ctr_x = ctr_x[:, np.newaxis] # 卧槽！卧槽！卧槽！卧槽！太流氓了，直接把anchor的x，就当做预测的x了，臭不要脸啊
                                      # 回答一下之前的注释的惊奇，这个是因为横坐标就是按照严格16个像素来的，其实就是anchor的x值
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes # 返回的是4个点的坐标

# 把超出的框的部分，都剪掉，比如超过右边界，就设为右边界坐标
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0  [0::4]表示从0开始，每隔4个的元素
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)  #im_shape[1] ， width
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)  #im_shape[0], height
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
