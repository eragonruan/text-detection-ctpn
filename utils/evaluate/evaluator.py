#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Algorithm named DetEval
It is slightly different from original algorithm(see https://perso.liris.cnrs.fr/christian.wolf/software/deteval/index.html)
Please read《 Object Count / Area Graphs for the Evaluation of Object Detection and Segmentation Algorithms 》for details

参考这篇；https://blog.csdn.net/weixin_35653315/article/details/71591596
'''
from collections import namedtuple
import importlib,math
import numpy as np


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


def evaluate_method(gt_points, detect_points, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row, j] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, j] >= \
                    evaluationParams['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i, col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[i, col] >= \
                    evaluationParams['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if (cont != 1):
            return False

        if recallMat[row, col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, col] >= \
                evaluationParams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def one_to_many_match(gtNum):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                if precisionMat[gtNum, detNum] >= evaluationParams['AREA_PRECISION_CONSTRAINT']:
                    many_sum += recallMat[gtNum, detNum]
                    detRects.append(detNum)
        if many_sum >= evaluationParams['AREA_RECALL_CONSTRAINT']:
            return True, detRects
        else:
            return False, []

    def many_to_one_match(detNum):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum, detNum] >= evaluationParams['AREA_RECALL_CONSTRAINT']:
                    many_sum += precisionMat[gtNum, detNum]
                    gtRects.append(gtNum)
        if many_sum >= evaluationParams['AREA_PRECISION_CONSTRAINT']:
            return True, gtRects
        else:
            return False, []

    def area(a, b):
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



    methodRecallSum = 0
    methodPrecisionSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0
    recall = 0
    precision = 0
    hmean = 0
    recallAccum = 0.
    precisionAccum = 0.
    gtRects = []
    detRects = []
    gtPolPoints = []
    detPolPoints = []
    gtDontCareRectsNum = []  # Array of Ground Truth Rectangles' keys marked as don't Care
    detDontCareRectsNum = []  # Array of Detected Rectangles' matched with a don't Care GT
    pairs = []
    evaluationLog = ""

    recallMat = np.empty([1, 1])
    precisionMat = np.empty([1, 1])

    for n in range(len(gt_points)):
        points = gt_points[n]

        # convert x1,y1,x2,y2,x3,y3,x4,y4 to xmin,ymin,xmax,ymax
        if len(points) == 8:
            points_tmp = np.array(points).reshape(4, 2)
            points_x = points_tmp[:, 0]
            points_y = points_tmp[:, 1]
            xmin = points_x[np.argmin(points_x)]
            xmax = points_x[np.argmax(points_x)]
            ymin = points_y[np.argmin(points_x)]
            ymax = points_y[np.argmax(points_y)]
            points = [xmin, ymin, xmax, ymax]
        gtRect = Rectangle(*points)
        gtRects.append(gtRect)
        gtPolPoints.append(points)

    evaluationLog += "GT rectangles: " + str(len(gtRects)) + (
    " (" + str(len(gtDontCareRectsNum)) + " don't care)\n" if len(gtDontCareRectsNum) > 0 else "\n")


    for n in range(len(detect_points)):
        points = detect_points[n]
        # print points
        detRect = Rectangle(*points)
        detRects.append(detRect)
        detPolPoints.append(points)
        if len(gtDontCareRectsNum) > 0:
            for dontCareRectNum in gtDontCareRectsNum:
                dontCareRect = gtRects[dontCareRectNum]
                intersected_area = area(dontCareRect, detRect)
                rdDimensions = ((detRect.xmax - detRect.xmin + 1) * (detRect.ymax - detRect.ymin + 1))
                if (rdDimensions == 0):
                    precision = 0
                else:
                    precision = intersected_area / rdDimensions
                if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT']):
                    detDontCareRectsNum.append(len(detRects) - 1)
                    break

    evaluationLog += "DET rectangles: " + str(len(detRects)) + (
    " (" + str(len(detDontCareRectsNum)) + " don't care)\n" if len(detDontCareRectsNum) > 0 else "\n")

    if len(gtRects) == 0:
        recall = 1
        precision = 0 if len(detRects) > 0 else 1

    if len(detRects) > 0:
        # Calculate recall and precision matrixs
        outputShape = [len(gtRects), len(detRects)]
        recallMat = np.empty(outputShape)
        precisionMat = np.empty(outputShape)
        gtRectMat = np.zeros(len(gtRects), np.int8)
        detRectMat = np.zeros(len(detRects), np.int8)
        for gtNum in range(len(gtRects)):
            for detNum in range(len(detRects)):
                rG = gtRects[gtNum]
                rD = detRects[detNum]
                intersected_area = area(rG, rD)
                rgDimensions = ((rG.xmax - rG.xmin + 1) * (rG.ymax - rG.ymin + 1))
                rdDimensions = ((rD.xmax - rD.xmin + 1) * (rD.ymax - rD.ymin + 1))
                recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else  intersected_area / rgDimensions
                precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

        # Find one-to-one matches
        evaluationLog += "Find one-to-one matches\n"
        for gtNum in range(len(gtRects)):
            for detNum in range(len(detRects)):
                if gtRectMat[gtNum] == 0 and detRectMat[
                    detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum:
                    match = one_to_one_match(gtNum, detNum)
                    if match is True:
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        normDist = center_distance(rG, rD)
                        normDist /= diag(rG) + diag(rD)
                        normDist *= 2.0
                        if normDist < evaluationParams['EV_PARAM_IND_CENTER_DIFF_THR']:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            recallAccum += evaluationParams['MTYPE_OO_O']
                            precisionAccum += evaluationParams['MTYPE_OO_O']
                            pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                        else:
                            evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(
                                detNum) + " normDist: " + str(normDist) + " \n"
        # Find one-to-many matches
        evaluationLog += "Find one-to-many matches\n"
        for gtNum in range(len(gtRects)):
            if gtNum not in gtDontCareRectsNum:
                match, matchesDet = one_to_many_match(gtNum)
                if match is True:
                    gtRectMat[gtNum] = 1
                    recallAccum += evaluationParams['MTYPE_OM_O']
                    precisionAccum += evaluationParams['MTYPE_OM_O'] * len(matchesDet)
                    pairs.append({'gt': gtNum, 'det': matchesDet, 'type': 'OM'})
                    for detNum in matchesDet:
                        detRectMat[detNum] = 1
                    evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"

        # Find many-to-one matches
        evaluationLog += "Find many-to-one matches\n"
        for detNum in range(len(detRects)):
            if detNum not in detDontCareRectsNum:
                match, matchesGt = many_to_one_match(detNum)
                if match is True:
                    detRectMat[detNum] = 1
                    recallAccum += evaluationParams['MTYPE_OM_M'] * len(matchesGt)
                    precisionAccum += evaluationParams['MTYPE_OM_M']
                    pairs.append({'gt': matchesGt, 'det': detNum, 'type': 'MO'})
                    for gtNum in matchesGt:
                        gtRectMat[gtNum] = 1
                    evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if len(detRects) > 0 else float(1)
        else:
            recall = float(recallAccum) / numGtCare
            precision = float(0) if (len(detRects) - len(detDontCareRectsNum)) == 0 else float(
                precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    evaluationLog += "Recall = " + str(recall) + "\n"
    evaluationLog += "Precision = " + str(precision) + "\n"

    methodRecallSum += recallAccum
    methodPrecisionSum += precisionAccum
    numGt += len(gtRects) - len(gtDontCareRectsNum)
    numDet += len(detRects) - len(detDontCareRectsNum)

    perSampleMetrics = {
        'precision': precision,
        'recall': recall,
        'hmean': hmean,
        'pairs': pairs,
        'recallMat': [] if len(detRects) > 100 else recallMat.tolist(),
        'precisionMat': [] if len(detRects) > 100 else precisionMat.tolist(),
        'gtPolPoints': gtPolPoints,
        'detPolPoints': detPolPoints,
        'gtDontCare': gtDontCareRectsNum,
        'detDontCare': detDontCareRectsNum,
        'evaluationParams': evaluationParams,
        'evaluationLog': evaluationLog
    }

    methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
    methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
    methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    return methodMetrics, perSampleMetrics
