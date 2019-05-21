class Config:

    # 训练用的参数
    EPS = 1e-14
    RPN_CLOBBER_POSITIVES = False
    RPN_NEGATIVE_OVERLAP = 0.3 # IoU小于这个值，就认为他是背景，在训练筛选某个anchor是否是样本的时候用
    RPN_POSITIVE_OVERLAP = 0.7 # IoU大约这个值，我们认为就是前景，在训练筛选某个anchor是否是样本的时候用
    RPN_FG_FRACTION = 0.5      # 正样本，FG是前景的意思，在整体样本中的比率，0.5=50%，一般样本是正样本
    RPN_BATCHSIZE = 1200       # 真实样本我观察在150个左右的大框，800+小框，所以，我保守的用1200，也就是600个小框正样本
    RPN_BBOX_INSIDE_WEIGHTS = (0, 1.0, 0, 1.0) # (x,y,dx,dy),之前作者都写成了[1,1,1,1]，不对，只保留y和dy
    RPN_POSITIVE_WEIGHT = -1.0 # 不知道干吗用的
    NETWORK_ANCHOR_NUM = 4     # 一个feature map上的点备选的anchor数量，原来是10，我觉得没必要，改成了4：11，16，23，33

    # 预测用的参数
    RPN_FG_POSSIBILITY = 0.95 # 概率大于多少认为是前景，这个只是在预测的时候用
    RPN_PRE_NMS_TOP_N = 12000 # 12000，施加NMS前算一下，只要前12000个，IoU大的
    RPN_POST_NMS_TOP_N = 1000 # 1000，这个施加了NMS后，不过目前不用了，因为没必要，我只要
    RPN_NMS_THRESH = 0.7      # NMS的阈值，用来判断，如果大于0.7的重合度，就认为这个框可以被丢弃掉了
    RPN_MIN_SIZE = 5          # 预测的宽和高，小于这个值，这个框就会被抛弃掉

    # 图像的最大的宽与高
    RPN_IMAGE_WIDTH  = 1200
    RPN_IMAGE_HEIGHT = 1600
