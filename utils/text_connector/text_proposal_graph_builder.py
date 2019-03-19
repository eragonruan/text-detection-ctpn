import numpy as np

from utils.text_connector.other import Graph
from utils.text_connector.text_connect_cfg import Config as TextLineCfg


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    # 返回某一个框对应的？??
    # 文本线构造算法
    # https://zhuanlan.zhihu.com/p/34757009
    # 文本线构造算法通过如下方式建立每个Anchor \text{box}_i 的 \text{pair}(\text{box}_i, \text{box}_j) ：
    #
    # 正向寻找：
    #
    # 沿水平正方向，寻找和 \text{box}_i 水平距离小于50的候选Anchor
    # 从候选Anchor中，挑出与 \text{box}_i 水平方向 \text{overlap}_v >0.7 的Anchor
    # 挑出符合条件2中Softmax score最大的 \text{box}_j
    # 再反向寻找：
    #
    # 沿水平负方向，寻找和 \text{box}_j 水平距离小于50的候选Anchor
    # 从候选Anchor中，挑出与 \text{box}_j 水平方向 \text{overlap}_v >0.7 的Anchor
    # 挑出符合条件2中Softmax score最大的 \text{box}_k
    # 最后对比 \text{score}_i 和 \text{score}_k :
    #
    # 如果 \text{score}_i >= \text{score}_k ，则这是一个最长连接，那么设置 \text{Graph}(i, j) = \text{True}
    # 如果 \text{score}_i < \text{score}_k ，说明这不是一个最长的连接（即该连接肯定包含在另外一个更长的连接中）。
    #
    # 以上摘自知乎的说明，下面是我的理解
    #
    # 就是从起始的proposal(i)往右找50个像素以内的，score(前景概率)最大的那个proposal(j)
    # 然后折返跑，同样的查找方式(找50像素score最大的)，再找到proposal(k)
    # 然后比较score(i)和score(k)，只有score(i)>=score(k)，才承认i->j是联通的。
    #
    # 恩，敲黑板！最后判断的是i-->j，跟中间的那个k鸟关系没有，k只是一个中间的过客而已。
    # 有2种情况会出现，就是k出现在i的左面或者是右面：（如果是i==k,肯定满足了）
    # 1. k在i的左面
    #           i------->j
    #        k<----------|
    #    1.1 如果score(i)>=score(k),说明i更像是前景，我最优，那我就确认我是个强连通
    #    1.2 如果score(i)<score(k) ,说明k更像是前景，那说明i->j不是最佳联通，是包含在别人之内，但我们现在只判断i的后续，既然我不是最优，我就放弃了
    #
    # 2. k在i的右面
    #        i---------->j
    #           k<-——----|
    #    2.1 如果score(i)>=score(k),说明i更像是前景，我最优，我包含了k，那我就确认我是个强连通
    #    2.2 如果score(i)<score(k) ,说明k更像是前景，那说明i->j肯定不是最佳联通了，我就放弃了
    # 1和2，里面让人纠结是1.2，感觉k->j是一个更好的强连通，为何不确定下来呢？这块我也有点困惑，我觉得可能是，现在只判断i和后续的谁发生关系把，之前的就交给前面循环的i来处理吧。
    #
    # ！！！对，这个就是折返跑的前半段，往右跑，不过跑的过程中，没有找最大的，只是找50个像素内，跟他垂直高度在0.7相交的那帮，找最大的，在调用他的函数的内部
    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        # 从box[0],就是小框的x1开始，往右50个像素
        for left in range(int(box[0]) + 1,
                          min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1,
                              self.im_size[1])):
            # boxes_table是所有的x为下标的数组，存着以这个x开头的box index的list
            # 所有adj_box_indices是一个数组
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices: # 对每一个这个left的x上开头的box index
                # 有点晕：adj_box_index是往右的50个像素内的每个点对应的某个proposals的index
                #       index 是某个proposal的index，传入的，可以查看外层调用
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    # 就是从起始的proposal(i)往右找50个像素以内的，score(前景概率)最大的那个proposal(j)
    # 然后折返跑，同样的查找方式(找50像素score最大的)，再找到proposal(k)
    # 然后比较score(i)和score(k)，只有score(i)>=score(k)，才承认i->j是联通的。
    #
    # 恩，敲黑板！最后判断的是i-->j，跟中间的那个k鸟关系没有，k只是一个中间的过客而已。
    # 有2种情况会出现，就是k出现在i的左面或者是右面：（如果是i==k,肯定满足了）
    # 1. k在i的左面
    #           i------->j
    #        k<----------|
    #    1.1 如果score(i)>=score(k),说明i更像是前景，我最优，那我就确认我是个强连通
    #    1.2 如果score(i)<score(k) ,说明k更像是前景，那说明i->j不是最佳联通，是包含在别人之内，但我们现在只判断i的后续，既然我不是最优，我就放弃了
    #
    # 2. k在i的右面
    #        i---------->j
    #           k<-——----|
    #    2.1 如果score(i)>=score(k),说明i更像是前景，我最优，我包含了k，那我就确认我是个强连通
    #    2.2 如果score(i)<score(k) ,说明k更像是前景，那说明i->j肯定不是最佳联通了，我就放弃了
    # 1和2，里面让人纠结是1.2，感觉k->j是一个更好的强连通，为何不确定下来呢？这块我也有点困惑，我觉得可能是，现在只判断i和后续的谁发生关系把，之前的就交给前面循环的i来处理吧。
    #
    # 对，这个函数就是折返跑的后半段，往左跑
    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    # 看，这个框和他是不是上下和他相交在0.7以上
    # 干嘛呢？就是说，这个框和你得高度上相仿，别错位太多，你还连个屁啊
    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1] # self.heights存在所有的proposal的高度，朝右的像素对应的proposal的高度
            h2 = self.heights[index2] # 被考察的proposal的高度
            # 比较左上点的y，找大的y
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1]) #proposal[x0,y0,x1,y1]
            # 比较右下角点的y，找小的y
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])

            # 我理解，在这个结果是用y方向交的距离，除以高度小的那个
            # 算是一种iou计算？
            # 为何不考虑x?
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    # text_proposals：x1,y1,x2,y2
    # im_size[H,W]
    # 返回的是一个方阵的"图"，图是个隐喻，其实TMD就是个方阵，每一维度是proposal的数量。
    # 这个图形象的可以参考：https://pic1.zhimg.com/80/v2-822f0709d3e30df470a8e17f09a25de0_hd.jpg
    # 表明，我到你是否联通，比如 第三行第四列为true，表明，第三个proposal到第四个proposal是联通的
    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1 # 所有框的高度

        # 按照原图宽度，即所有x，建立一个数组
        boxes_table = [[] for _ in range(self.im_size[1])] # im_size:h,w,原图的
        for index, box in enumerate(text_proposals):
            # 如果某个x上有值，就把对应的proposal加到数组里
            #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* <----   * 是x的像素点
            #           |_[index11]     |_[index14]     index??是proposal的索引
            #           |_[index13]
            #           |_[index31]
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        # graph是一个proposal数量的方阵，false都是
        # 这个图形象的可以参考：https://pic1.zhimg.com/80/v2-822f0709d3e30df470a8e17f09a25de0_hd.jpg
        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        # 遍历所有的proposal，也就是候选框
        for index, box in enumerate(text_proposals):

            # 在干嘛？在找和他纵向IoU在0.7的框，50个像素以内的
            # 返回的是一个索引，是proposal的索引
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue

            # 然后按照分数大小有排了个序
            # 然后找到了一个成绩(score成绩，最前景的)最大的那个proposal的"索引index"
            # ！！！对，这是文本识别中的折返跑的前半段，往右跑的时候，找那个最大的j
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors)
                # if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)
