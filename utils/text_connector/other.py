import numpy as np

import logging

logger  = logging.getLogger("sub_graphs")

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes

# graph是proposalxproposal的方阵
# 来表明我和你之间是否联通
class Graph:
    def __init__(self, graph):
        self.graph = graph

    # 返回一堆的列表，每个列表就是一嘟噜的bbox，说明他们连通着
    def sub_graphs_connected(self):

        logger.debug("Graph.shape:%r",self.graph.shape)
        # for row in self.graph:
        #     for col in row:
        #         self.x = print(col, end='')
        #     print()

        sub_graphs = []
        # 遍历每一行，记住，graph是proposalxproposal的方阵
        for index in range(self.graph.shape[0]):

            # 2019.5.28 piginzoo 这行表达的是，这个bbox既不指向别人，也不被别人指着，过去这种框就被忽略了，
            # 这是个问题，这种单独的框会被忽略，这种我要留下来，修正这个bug
            if not self.graph[:, index].any() and not self.graph[index, :].any():
                sub_graphs.append([index])
                continue

            # any函数，意思是numpy数组中其中有一个元素是true，我结果就返回true
            # graph[:, index].any()，看谁和我联通
            # graph[index, :].any()，看我和谁联通
            # 表示，我和别人联通，别人也和我联通
            # 我是指当前的index
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index

                # 那把我加到sub_grophs里面，v是行号哈
                # 注意注意，append的是[v]，不是v
                # 啥意思？
                # 就是sub_graph是一个二维的数组，数组里面套数组，每一个行都是这个index
                sub_graphs.append([v])

                # 这个循环在干嘛？
                # "graph[v, :].any()"，表示，我和别人联通
                while self.graph[v, :].any():
                    # https://www.cnblogs.com/massquantity/p/8908859.html
                    # a = np.array([[1.1, 1.1, 1.1], [2.2, 2.2, 2.2], [3.3, 3.3, 3.3], [4.4, 4.4, 4.4]])
                    # np.where(a[1:])
                    # (array([0, 0, 0, 1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
                    # where返回的是他开头->某个结尾的那个结尾，就是v，这个需要充分理解where
                    # logger.debug("np.where(self.graph[v, :])%r",np.where(self.graph[v, :]))
                    # 表示，找到我联通的下一个人的索引v
                    v = np.where(self.graph[v, :])[0][0]

                    # print("%d->" % v,end='')
                    # 结束条件是v找不到下一个联通点了，while就退出了
                    sub_graphs[-1].append(v)
                # print()

        # 返回的是一个list[list]
        # 如：[[1-2-4-6-9],
        #     [2-3-5-11],
        #     [7-13-14-15]]
        # 每一行是一个list，是一个联通的index的list
        logger.debug("找到联通子图sub_graphs: %d",len(sub_graphs))
        return sub_graphs