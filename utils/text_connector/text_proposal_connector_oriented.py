# coding:utf-8
import numpy as np

from utils.text_connector.text_proposal_graph_builder import TextProposalGraphBuilder


class TextProposalConnector:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    # im_size[H,W]
    def group_text_proposals(self, text_proposals, scores, im_size):

        # build_graph 返回了一个图，参见：https://pic1.zhimg.com/80/v2-822f0709d3e30df470a8e17f09a25de0_hd.jpg
        # 是一个proposal（排序了+NMS之后的）的数量组成的方阵，用来标明谁和谁联通的
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        # 返回的是一个list[list]
        # 如：[[1-2-4-6-9],
        #     [2-3-5-11],
        #     [7-13-14-15]]
        # 每一行是一个list，是一个联通的index的list
        return graph.sub_graphs_connected()

    # X，Y是要被拟合的小框的左上角的X，Y，或者是右下角的
    # x1是最左面点的x + 框宽度的一半（一半难道是8？）
    # x2是最右面点的x - 框宽度的一半
    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        # 这个是说，通过X, Y，拟合出一条曲线，polyfit第三个参数是1，说明是一次曲线，即直线
        # 然后用这个直线（polyfit返回的是y=kx+b的k,b)，去算出最左面点和最右面的点
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    # 为了得到文本框，大框，返回的是拼在一起的大框的4个点的坐标
    # text_proposals：按照前景的概率从大到小排序了的；通过NMS去除了次优的；
    # scores：就是对应的前景的置信度/概率
    # im_size：只保留了高和宽[H,W]
    def get_text_lines(self, text_proposals, scores, im_size): #im_size:H,W
        """
        text_proposals:boxes
        
        """
        # group_text_proposals得到的的是一个list[list]
        # 如：[[1-2-4-6-9],
        #     [2-3-5-11],
        #     [7-13-14-15]]
        # 每一行是一个list，是一个联通的index的list
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        # tp_groups每一行是一嘟噜联通的小的proposal的index
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框

            # text_line_boxes[x1,y1,x2,y2]
            # X,Y得到的是一个数组，一排X和Y
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            # https://drivingc.com/numpy/5af5ab892392ec35c23048e2
            # 找一条直线，因为第三个参数是1，即degree=1，只做直线拟合
            # 得到的z1是k和b，即y=kx+b
            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            # （第一个小框的x2 - 第一个小框的x1）* 0.5，恩，第一个小框的宽度的一半
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x4
            text_recs[index, 5] = y4
            text_recs[index, 6] = x3
            text_recs[index, 7] = y3
            text_recs[index, 8] = line[4]
            index = index + 1

        # 返回的是拼在一起的大框的4个点的坐标
        return text_recs
