class Config:
    MAX_HORIZONTAL_GAP = 20 # 两个大框之间最短的距离
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.2 # 这个值很小，是因为，剔除和他稍微挨着的，但是置信度没有他高的框，防止重叠度太高
    MIN_V_OVERLAPS = 0.7
    MIN_SIZE_SIM = 0.7
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9 # 0.9=>0.7 我给改小了，从0.9->0.7，原因是发现有些识别不出来
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
