class Config:

    # 本地调试用
    # EPS = 1e-14
    # RPN_CLOBBER_POSITIVES = False
    # RPN_NEGATIVE_OVERLAP = 0.3
    # RPN_POSITIVE_OVERLAP = 0.7
    # RPN_FG_FRACTION = 0.5
    # RPN_BATCHSIZE = 8
    # RPN_BBOX_INSIDE_WEIGHTS = (0, 1.0, 0, 1.0) # (x,y,dx,dy),之前作者都写成了[1,1,1,1]，不对，只保留y和dy
    # RPN_POSITIVE_WEIGHT = -1.0
    #
    # RPN_PRE_NMS_TOP_N = 1200
    # RPN_POST_NMS_TOP_N = 100
    # RPN_NMS_THRESH = 0.7
    # RPN_MIN_SIZE = 8

    # production evn
    EPS = 1e-14
    RPN_CLOBBER_POSITIVES = False
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 600
    RPN_BBOX_INSIDE_WEIGHTS = (0, 1.0, 0, 1.0) # (x,y,dx,dy),之前作者都写成了[1,1,1,1]，不对，只保留y和dy
    RPN_POSITIVE_WEIGHT = -1.0

    RPN_PRE_NMS_TOP_N = 12000
    RPN_POST_NMS_TOP_N = 1000
    RPN_NMS_THRESH = 0.7
    RPN_MIN_SIZE = 8
