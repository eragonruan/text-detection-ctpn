#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Algorithm named DetEval
It is slightly different from original algorithm(see https://perso.liris.cnrs.fr/christian.wolf/software/deteval/index.html)
Please read《 Object Count / Area Graphs for the Evaluation of Object Detection and Segmentation Algorithms 》for details

参考这篇；https://blog.csdn.net/weixin_35653315/article/details/71591596

ICDAR2013则使用了新evaluation方法：DetEval，也就是十几年前Wolf提出的方法。“新方法”同时考虑了一对一，一对多，多对一的情况， 但不能处理多对多的情况。
（作者说，实验结果表示在文本检测里这种情况出现的不多。）

这里的框无论是标定框还是检测框都认为是水平的矩形框

'''
from collections import namedtuple
import importlib,math
import numpy as np
import logging

logger = logging.getLogger("Evaluator")

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
Point = namedtuple('Point', 'x y')
recall = 0
precision = 0
hmean = 0
gtRects = []
detRects = []
pairs = []
evaluationLog = ""

def conf():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT': 0.8,
        'AREA_PRECISION_CONSTRAINT': 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O': 1.,
        'MTYPE_OM_O': 0.8,
        'MTYPE_OM_M': 1.,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'CRLF': False  # Lines are delimited by Windows CRLF format
    }

# row是GT，col是Det
# 1对1，参考：https://blog.csdn.net/liuxiaoheng1992/article/details/82632594
# 召回率：大于0.8, 交/GT
# 精确率：大于0.4, 交/Det
# GT找到符合条件的只有1个Det
# Det找到符合条件的只有1个GT
# 而且这个GT和Det就是传入的row\col对应的那个
def one_to_one_match(row, col,conf,recallMat,precisionMat):
    cont = 0
    # 看recall矩阵里面有没有大于大于阈值r(0.8)和p(0.4)的
    # 就是说，看看有没有跟某个GT相交程度大于40%的
    for j in range(len(recallMat[0])):
        if recallMat[row, j] >= conf['AREA_RECALL_CONSTRAINT'] and \
        precisionMat[row, j] >= conf['AREA_PRECISION_CONSTRAINT']:
            cont = cont + 1
    # 如果有多个，那么你就不是1对1
    if (cont != 1):
        return False
    cont = 0
    # 然后再看，要验证的那个列（探测框）大于阈值r(0.8)和p(0.4)
    for i in range(len(recallMat)):
        if recallMat[i, col] >= conf['AREA_RECALL_CONSTRAINT'] and \
        precisionMat[i, col] >= conf['AREA_PRECISION_CONSTRAINT']:
            cont = cont + 1
    if (cont != 1):
        return False

    # 看看，从recall矩阵中找的，和从precision矩阵中找的备选点，是不是一个点
    if recallMat[row, col] >= conf['AREA_RECALL_CONSTRAINT'] and \
    precisionMat[row, col] >= conf['AREA_PRECISION_CONSTRAINT']:
        return True

    return False

#1个GT对多个Det探测框：https://blog.csdn.net/liuxiaoheng1992/article/details/82632594
def one_to_many_match(gtNum,conf,gtRectMat,detRectMat,recallMat,precisionMat):
    many_sum = 0
    detRects = []
    for detNum in range(len(recallMat[0])):
        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 :
            if precisionMat[gtNum, detNum] >= conf['AREA_PRECISION_CONSTRAINT']:
                many_sum += recallMat[gtNum, detNum]
                detRects.append(detNum)
    if many_sum >= conf['AREA_RECALL_CONSTRAINT']:
        return True, detRects
    else:
        return False, []

# 多个GT，对应一个Det（探测框）：https://blog.csdn.net/liuxiaoheng1992/article/details/82632594
def many_to_one_match(detNum,conf,gtRectMat,detRectMat,recallMat,precisionMat):
    many_sum = 0
    gtRects = []
    for gtNum in range(len(recallMat)):
        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 :
            if recallMat[gtNum, detNum] >= conf['AREA_RECALL_CONSTRAINT']:
                many_sum += precisionMat[gtNum, detNum]
                gtRects.append(gtNum)
    if many_sum >= conf['AREA_PRECISION_CONSTRAINT']:
        return True, gtRects
    else:
        return False, []

# 算a，b代表的4边形的相交面积，ab都是矩形，
# xmax是右侧的x，xmin是左侧的x，ymax是下边的y，ymin是上边的y
# 求出来的是a、b两个矩形，相交的面积
def area(a, b):
    #
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0.

def center(r):
    x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
    y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
    return Point(x, y)

def point_distance(r1, r2):
    distx = math.fabs(r1.x - r2.x)
    disty = math.fabs(r1.y - r2.y)
    return math.sqrt(distx * distx + disty * disty)

def center_distance(r1, r2):
    return point_distance(center(r1), center(r2))

def diag(r):
    w = (r.xmax - r.xmin + 1)
    h = (r.ymax - r.ymin + 1)
    return math.sqrt(h * h + w * w)

def evaluate(gt_points, detect_points, conf):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    # 得到所有的GT坐标，注意，这里的GT不是split的小框，是大框，即8个值，4个坐标点，就是对应的4边形的4个顶点
    for n in range(len(gt_points)):
        points = gt_points[n]

        # logger.debug(points)
        # convert x1,y1,x2,y2,x3,y3,x4,y4 to xmin,ymin,xmax,ymax
        if len(points) >= 8:
            points_tmp = np.array(points).reshape(4, 2)
            points_x = points_tmp[:, 0]
            points_y = points_tmp[:, 1]
            xmin = points_x[np.argmin(points_x)] # 这里，把不规则的4边型，变成矩形
            xmax = points_x[np.argmax(points_x)]
            ymin = points_y[np.argmin(points_x)]
            ymax = points_y[np.argmax(points_y)]
            points = [xmin, ymin, xmax, ymax]
        else:
            points = points[:4] # 4个坐标的，只保留前4个值，最后一个置信度扔掉

        gtRect = Rectangle(*points)
        gtRects.append(gtRect)
        # gtPolPoints.append(points)

    logger.debug("一共有样本框（GT）:%d", len(gt_points))

    # 这个是探测的点，记住，是CTPN合并后的矩形的4个点
    for n in range(len(detect_points)):
        points = detect_points[n]
        # convert x1,y1,x2,y2,x3,y3,x4,y4 to xmin,ymin,xmax,ymax
        if len(points) == 8:
            points_tmp = np.array(points).reshape(4, 2)
            points_x = points_tmp[:, 0]
            points_y = points_tmp[:, 1]
            xmin = points_x[np.argmin(points_x)] # 这里，把不规则的4边型，变成矩形
            xmax = points_x[np.argmax(points_x)]
            ymin = points_y[np.argmin(points_x)]
            ymax = points_y[np.argmax(points_y)]
            points = [xmin, ymin, xmax, ymax]
        # print points
        detRect = Rectangle(*points)
        detRects.append(detRect)

    logger.debug("一共有预测的框 :%d", len(detect_points))

    '''
    recall与precision两个矩阵,这两个矩阵合在一起也称为overlap matrices. i,j位置上的元素值不为0就代表GT_i,Det_j 之间有重合。
    '''
    # Calculate recall and precision matrixs
    outputShape = [len(gtRects), len(detRects)]
    recallMat =    np.empty(outputShape)
    precisionMat = np.empty(outputShape)
    gtRectMat =    np.zeros(len(gtRects),  np.int8) # gtRectMat这个用来标识我是否已经被识别过了，省的再用它了，我理解是为了提高效率
    detRectMat =   np.zeros(len(detRects), np.int8) #
    recallAccum = 0. # 这个是为了算所有的recall值，是一个累计值，最后要用它除以
    precisionAccum = 0.

    # 构建 召回率矩阵 和 精确率矩阵
    for gtNum in range(len(gtRects)):
        for detNum in range(len(detRects)):
            rG = gtRects[gtNum]   # 某一个gt框
            rD = detRects[detNum] # 某一个探测的框
            intersected_area = area(rG, rD) # 算他们的相交面积
            rgDimensions = ((rG.xmax - rG.xmin + 1) * (rG.ymax - rG.ymin + 1)) # GT的面积
            rdDimensions = ((rD.xmax - rD.xmin + 1) * (rD.ymax - rD.ymin + 1)) # 探测框的面积
            # 召回率：真正例/所有样本正例
            recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else  intersected_area / rgDimensions
            # 精确率：真正例/所有预测正例
            precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

    # ####################################################################################
    #    one-to-one matches  1:1
    # ####################################################################################

    # 在recallMat和precisionMat中的i行只有一个值大于阈值，j列中也只有一个值大于阈值，且这个值在第i行第j列，
    # 那么就认为gt_i与det_j是one-to-one matches
    # 大白话：就是我(i,gt)只和你(j,det)相交，你也只和我相交 （所谓相交，就是大于某个阈值）
    one2one = 0
    for gtNum in range(len(gtRects)):
        for detNum in range(len(detRects)):
            # gtRectMat这个用来标识我是否已经被识别过了，省的再用它了，我理解是为了提高效率
            if gtRectMat[gtNum] != 0 or detRectMat[detNum] != 0: continue

            match = one_to_one_match(gtNum, detNum,conf,recallMat,precisionMat)
            if not match: continue

            rG = gtRects[gtNum]   # 恩，我(GT)
            rD = detRects[detNum] # 恩， 你（Det），你我都是个矩形
            normDist = center_distance(rG, rD) # 算算我们的中心点的距离
            normDist /= diag(rG) + diag(rD)    #
            normDist *= 2.0
            # 还要满足个条件：
            # 两个框的中心点距离与两个框对角线平均值的比例要小于阈值1
            if normDist < conf['EV_PARAM_IND_CENTER_DIFF_THR']:
                # logger.debug("找到一个1:1的框")
                gtRectMat[gtNum] = 1   # 设这个gt框的recall=1
                detRectMat[detNum] = 1 # 设这个det框的percision=1
                recallAccum += conf['MTYPE_OO_O']    # 算你一个
                precisionAccum += conf['MTYPE_OO_O'] # 算你一个
                pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                one2one+=1
            else:
                logger.debug("虽然匹配但是不满足中心点距离与两个框对角线平均值的比例")
    logger.debug("1:1一共%d个", one2one)

    # ####################################################################################
    #    One-to-Many matches  1:M
    # ####################################################################################

    # one是指GT，many是指Det探测
    # 对于precisionMat中如果i行(对应某个GT）中有多个值大于p(0.4)，将对应于recallMat位置的值相加,注意是recall矩阵
    # 然后你把这个行里面的recall值都加起来，看看是否大于r(0.8)，
    # 这个时候，把这个框如果满足one-to-many matches就将recall加0.8，precision加0.8×num，
    # num表示对应与gt_i匹配的所有many框的个数(说白了就是many的具体值)
    logger.debug("开始探测1:m的")
    m2one = 0
    for gtNum in range(len(gtRects)):

        match, matchesDet = one_to_many_match(gtNum,conf,gtRectMat,detRectMat,recallMat,precisionMat)
        if not match: continue
        gtRectMat[gtNum] = 1
        m2one += 1
        recallAccum += conf['MTYPE_OM_O'] # 就将recall加0.8
        precisionAccum += conf['MTYPE_OM_O'] * len(matchesDet) # precision加0.8×num
        pairs.append({'gt': gtNum, 'det': matchesDet, 'type': 'OM'})
        for detNum in matchesDet:
            detRectMat[detNum] = 1
    logger.debug("1:m一共%d个",m2one)

    # ####################################################################################
    #    many-to-one matches  m:1
    # ####################################################################################

    logger.debug("开始探测m:1的")
    one2m = 0
    for detNum in range(len(detRects)):
        match, matchesGt = many_to_one_match(detNum,conf,gtRectMat,detRectMat,recallMat,precisionMat)
        if not match: continue
        one2m+=1
        detRectMat[detNum] = 1
        recallAccum += conf['MTYPE_OM_M'] * len(matchesGt)
        precisionAccum += conf['MTYPE_OM_M']
        pairs.append({'gt': matchesGt, 'det': detNum, 'type': 'MO'})
        for gtNum in matchesGt:
            gtRectMat[gtNum] = 1
    logger.debug("m:1一共%d个",m2one)

    logger.debug("一共%d个探测框匹配不上任何GT(r>0.8/p>0.4)",np.sum(gtRectMat))

    # 用recall除以所有的gt个数
    recall = float(recallAccum) / len(gtRects)
    # precision也会除以所有的det的个数
    precision =  float(precisionAccum) / len(detRects)
    hmean =  2.0 * precision * recall / (precision + recall)

    result = {
        'precision': precision,
        'recall': recall,
        'hmean': hmean,
        # 'pairs': pairs,
    }

    return result
