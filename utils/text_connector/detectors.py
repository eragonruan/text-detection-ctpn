# coding:utf-8
import numpy as np
from nms import nms

from .text_connect_cfg import Config as TextLineCfg
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented

import logging

logger = logging.getLogger("text detector")

class TextDetector:
    def __init__(self, DETECT_MODE="H"):
        self.mode = DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, size):

        logger.debug("detect - text proposal shape:%r",text_proposals.shape)
        logger.debug("detect - score:%r", scores.shape)
        logger.debug("detect - size:%r", size)

        # 删除得分较低的proposal,TEXT_PROPOSALS_MIN_SCORE=0.7,这个是前景的置信度
        keep_inds = np.where(scores > TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        logger.debug("detect - keep_inds:%d", len(keep_inds))
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]


        # 按得分排序，按照得分对text_proposals进行了排序，argsoft是从小到大，[::-1]是再反过来，即从大到小
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        # nms就是刨除掉那些比最优次优的proposals，
        # 当然如果差很多就不刨除，反倒保留着，根据阈值
        keep_inds = nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 获取检测结果
        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        keep_inds = self.filter_boxes(text_recs)
        return text_recs[keep_inds]

    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1

        return np.where((widths / heights > TextLineCfg.MIN_RATIO) & (scores > TextLineCfg.LINE_MIN_SCORE) &
                        (widths > (TextLineCfg.TEXT_PROPOSALS_WIDTH * TextLineCfg.MIN_NUM_PROPOSALS)))[0]
