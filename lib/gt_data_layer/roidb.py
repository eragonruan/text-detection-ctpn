# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
import scipy.sparse
import PIL
import math
import os
import cPickle
import pdb

from ..utils.cython_bbox import bbox_overlaps
from ..utils.boxes_grid import get_boxes_grid
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
# <<<< obsolete


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    cache_file = os.path.join(imdb.cache_path, imdb.name + '_gt_roidb_prepared.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            imdb._roidb = cPickle.load(fid)
        print '{} gt roidb prepared loaded from {}'.format(imdb.name, cache_file)
        return

    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        boxes = roidb[i]['boxes']
        labels = roidb[i]['gt_classes']
        info_boxes = np.zeros((0, 18), dtype=np.float32)

        if boxes.shape[0] == 0:
            roidb[i]['info_boxes'] = info_boxes
            continue

        # compute grid boxes
        s = PIL.Image.open(imdb.image_path_at(i)).size
        image_height = s[1]
        image_width = s[0]
        boxes_grid, cx, cy = get_boxes_grid(image_height, image_width)
        
        # for each scale
        for scale_ind, scale in enumerate(cfg.TRAIN.SCALES):
            boxes_rescaled = boxes * scale

            # compute overlap
            overlaps = bbox_overlaps(boxes_grid.astype(np.float), boxes_rescaled.astype(np.float))
            max_overlaps = overlaps.max(axis = 1)
            argmax_overlaps = overlaps.argmax(axis = 1)
            max_classes = labels[argmax_overlaps]

            # select positive boxes
            fg_inds = []
            for k in xrange(1, imdb.num_classes):
                fg_inds.extend(np.where((max_classes == k) & (max_overlaps >= cfg.TRAIN.FG_THRESH))[0])

            if len(fg_inds) > 0:
                gt_inds = argmax_overlaps[fg_inds]
                # bounding box regression targets
                gt_targets = _compute_targets(boxes_grid[fg_inds,:], boxes_rescaled[gt_inds,:])
                # scale mapping for RoI pooling
                scale_ind_map = cfg.TRAIN.SCALE_MAPPING[scale_ind]
                scale_map = cfg.TRAIN.SCALES[scale_ind_map]
                # contruct the list of positive boxes
                # (cx, cy, scale_ind, box, scale_ind_map, box_map, gt_label, gt_sublabel, target)
                info_box = np.zeros((len(fg_inds), 18), dtype=np.float32)
                info_box[:, 0] = cx[fg_inds]
                info_box[:, 1] = cy[fg_inds]
                info_box[:, 2] = scale_ind
                info_box[:, 3:7] = boxes_grid[fg_inds,:]
                info_box[:, 7] = scale_ind_map
                info_box[:, 8:12] = boxes_grid[fg_inds,:] * scale_map / scale
                info_box[:, 12] = labels[gt_inds]
                info_box[:, 14:] = gt_targets
                info_boxes = np.vstack((info_boxes, info_box))

        roidb[i]['info_boxes'] = info_boxes

    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb prepared to {}'.format(cache_file)

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'info_boxes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['info_boxes']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 12] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 14:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 14:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['info_boxes']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 12] == cls)[0]
            roidb[im_i]['info_boxes'][cls_inds, 14:] -= means[cls, :]
            if stds[cls, 0] != 0:
                roidb[im_i]['info_boxes'][cls_inds, 14:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image. The targets are scale invariance"""

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((ex_rois.shape[0], 4), dtype=np.float32)
    targets[:, 0] = targets_dx
    targets[:, 1] = targets_dy
    targets[:, 2] = targets_dw
    targets[:, 3] = targets_dh
    return targets
