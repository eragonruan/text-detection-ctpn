from .text_connect_cfg import Config as TextLineCfg
from .other import Graph
import numpy as np


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index):
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

    def get_precursors(self, index):
        box=self.text_proposals[index]
        results=[]
        for left in range(int(box[0])-1, max(int(box[0]-TextLineCfg.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals=text_proposals
        self.scores=scores
        self.im_size=im_size
        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1

        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        graph=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            succession_index=successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index]=True
                        
        # delete the one that are continous decline or rise       
        for index in range(graph.shape[0]):
            if graph[index, :].any():
                succession_index = np.where(graph[index, :])[0][0]
                if graph[succession_index, :].any():
                    next_succession_index = np.where(graph[succession_index, :])[0][0]
                    if not self.meet_v_iou(index, next_succession_index):
                        graph[index, succession_index]=False

        return Graph(graph)
