import numpy as np

# 我觉得这个代码是当年Faster-RCNN的anchor产生用的，复用了一下
#
def generate_basic_anchors(sizes, base_size=16):
    #奇怪，都是从0，0开始的？为何？
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors

# 看，返回的是anchor的x1,y1,x2,y2,不是x,y,w,h，这点要留意
def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor

# CTPN为fc feature map每一个点都配备10个上述Anchors
# https://zhuanlan.zhihu.com/p/34757009
# 返回：10个不同高度的anchor的4个坐标，是一个 10x4 的数组
# scale：8,16,32
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed;

    embed()
