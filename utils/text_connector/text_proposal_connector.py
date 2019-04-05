import numpy as np

from utils.text_connector.other import clip_boxes
from utils.text_connector.text_proposal_graph_builder import TextProposalGraphBuilder

# Horizental 连接器，水平连接器
class TextProposalConnector:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1)) # 用x，y来拟合一个直线
        return p(x1), p(x2) # 用这个拟合直线获得y值

    # 这个函数是把穿起来的text_proposals，一嘟噜，一串，串成一个四边形
    # text_proposals:[[x11,y11,x12,y12],[x21,y21,x22,y22],...]
    # ===> [x1,y1,x2,y2,x3,y3,x4,y4]
    # 返回的是，是一个list，每个list的元素是一个9个值的四边形(8个点)和1个score值
    def get_text_lines(self, text_proposals, scores, im_size):
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)

        # tp_groups 是所有的嘟噜，是一个list
        # 里面每一个元素，又是一嘟噜，是一个list
        # tp_groups=[
        #    [[x11,y11,x12,y12],[x21,y21,x22,y22]],
        #    ...
        # ]
        for index, tp_indices in enumerate(tp_groups):
            # text_line_boxes = [[x11,y11,x12,y12],[x21,y21,x22,y22]],
            text_line_boxes = text_proposals[list(tp_indices)]

            # 找到这一嘟噜的左面盒子的x0，和右面的盒子的x1，也就是得到这个一嘟噜的左右的边界
            x0 = np.min(text_line_boxes[:, 0]) # 0 is x_0 , text_line_boxes[[x1,y1,x2,y2],....]
            x1 = np.max(text_line_boxes[:, 2]) # 2 is x_1

            # x2-x1，第一个框的0，他的中心点，所以offset是第一个框的中心点
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            # [x0,y0],[x1,y1]
            #   0  1    2  3
            # (x0,y0)
            #   +------------+
            #   |            |
            #   +------------+(x1,y1)
            # x0,y0 - 左上角，通过一系列每个小框的左上角的点，拟合出一条直线，然后求直线两头的2个点的y
            # lt= left top
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # x0,y1 - 左下角，通过一系列每个小框的左下角的点，拟合出一条直线，然后求直线两头的2个点的y
            # lb = left bottom
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            # 你看！text_lines就是一个长方形啊，只有x0,y0,x1,y1和score
            # score，是大家的score的平均值
            # ???奇怪，为何整成了长方形，其实，完全可以搞成4个点的啊?:
            # text_lines[index, 0] = x0
            # text_lines[index, 1] = lt_y
            # text_lines[index, 2] = x1
            # text_lines[index, 3] = rt_y
            # text_lines[index, 4] = x1
            # text_lines[index, 5] = rb_y
            # text_lines[index, 6] = x0
            # text_lines[index, 7] = lb_y
            # text_lines[index, 4] = score
            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score

        text_lines = clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0

        # 这个其实没干啥，就是把上面弄出来的4个点，也就是矩形，变成8个点
        # 为何为何？为何不直接用8个点，为何，矩形就那么好么？
        # 不过后面算evaluate的时候，倒是方便了，因为都是矩形了。。。
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmax
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmin
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
