import numpy as np
from .config import cfg
from ..utils.nms import nms
from lib.utils.gpu_nms import gpu_nms

def nms(dets, thresh,use_gpu=True):
    if dets.shape[0] == 0:
        return []
    if use_gpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return nms(dets, thresh)
