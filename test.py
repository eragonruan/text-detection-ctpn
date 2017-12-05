import numpy as np
from lib.utils.bbox import bbox_overlaps
boxes=np.array([[0,0,1,1],[0,0,2,2]],np.float).reshape([-1,4])
query_boxes=np.array([[0,0,2,2],[0,0,0,0]],np.float).reshape([-1,4])
ov=bbox_overlaps(boxes, query_boxes)
print(ov)
