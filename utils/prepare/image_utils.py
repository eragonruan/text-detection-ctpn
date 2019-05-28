import numpy as np
from shapely.geometry import Polygon
import cv2,logging

IGNORE_WIDTH = 2 # 小于等于5像素的框，就忽略掉了，太小了，没意义

logger = logging

# 就是把左上点调整到数组的第一个
def pickTopLeft(poly):
    # 按照第一列，也就是x，进行排序，从小到大排，最左面在前面
    idx = np.argsort(poly[:, 0])
    # 比较最左面的第2个点的y>第1个点的y
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0] # 找到左上角，也就是用最靠左面的2个点，找那个最上面的点，s是对应的point的行号
    else:
        s = idx[1]

    # 返回的是
    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]

# 这个方法大概读了一下，就是返回闭包的四边形，
# 结果也不一定是矩形或者平行四边形，感觉是凸四边形
def orderConvex(p):
    points = Polygon(p).convex_hull # 返回最小凸包点
    points = np.array(points.exterior.coords)[:4]  # 获取多边形的外环坐标 polygon.exterior.coords 、 https://blog.csdn.net/wiborgite/article/details/85167397
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points

#poly还是shape(4,2)的四边形，不规则的可能是
# 每隔16像素，产生一个小的四边形，返回的是这个4变形的4个坐标
def shrink_poly(poly, r=16):
    print("处理一个大框")

    # 找最小的x
    x_min = int(np.min(poly[:, 0]))
    # 最大的x
    x_max = int(np.max(poly[:, 0]))

    # 上边的斜率和解决
    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]
    # 下边的斜率和解决
    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    # 存着切分完的矩形，应该是4个坐标，x1,y1,x2,y2
    res = []

    # # 取整一下吧
    # start = int((x_min // 16 + 1) * 16)
    # end = x_max   # int((x_max // 16) * 16) #我没有取整16
    # p = x_min
    # res.append([p, int(k1 * p + b1), # kx+b, p相当于是x，第一个p是最左面，但是第二个p就是16位取整的值了
    #             start - 1, int(k1 * (p + 15) + b1),
    #             start - 1, int(k2 * (p + 15) + b2),
    #             p, int(k2 * p + b2)])
    # 2009.5.27 piginzoo，注释上面的代码，为了是让左面的框，如果不是太窄(3个像素)，都直接扩到16个像素宽
    start = int((x_min // 16) * 16)
    end = x_max   # int((x_max // 16) * 16) #我没有取整16
    if (start + 16 - x_min) <= IGNORE_WIDTH:
        print("左面的框太小，忽略从往右的下一个位置开始：%d" % (start + 16 - x_min))
        start = start + 16


    for p in range(start, end + 1, r):
        # 2019.4.7 我给改成了收紧，让小框紧紧的包裹住大框的右侧边缘
        # 后来觉得自作聪明了，因为，这样会让右面出现一个特别窄的小框
        # 左面这样倒也罢了，毕竟是因为没办法，因为要和16像素对齐，must be aligned with 16pxs
        # 可右面，你还整这么窄，不是自我zuo么？！
        # 赶紧去掉
        # if (end-p) < 16:  <-----之前自作聪明的修改，打脸啊
        #     right = end
        # else:
        # 2019.5.10 再次修改，右面不是直接+16，而是看，如果我和end的距离2个像素内，那么这个GT框我就不算了
        #     right = p + 16
        right = p + 15
        if (end - p)<= IGNORE_WIDTH:
            print("宽度太小，最后框不要：%d" % (end - p))
            continue


        # 左上，右上，右下，坐下 => [x1,y1,x2,y2,x3,y3,x4,y4]
        # 看，res是一个四边形，不是一个矩形
        res.append([p,                      # 上方的x1
                    int(k1 * p + b1),       # 上方的y1

                    right,               # 上方的x2
                    int(k1 * (right) + b1),# 上方的y2

                    right,               # 下方的x3
                    int(k2 * (right) + b2),# 下方的y3

                    p,                      # 下方的x4
                    int(k2 * p + b2)])      # 下方的y4

    return np.array(res, dtype=np.int).reshape([-1, 8])

# 看哪个大了，就缩放哪个，规定最大的宽和高：max_width,max_height
def resize_image(image,max_width,max_height):
    h,w,_ = image.shape # H,W

    if h<max_height and w<max_width:
        logger.debug("图片的宽高[%d,%d]比最大要求[%d,%d]小，无需resize",h,w,max_height,max_width)
        return image,1

    h_scale = max_height/h
    w_scale = max_width/w
    # print("h_scale",h_scale,"w_scale",w_scale)
    scale = min(h_scale,w_scale) # scale肯定是小于1的，越小说明缩放要厉害，所以谁更小，取谁

    # https://www.jianshu.com/p/11879a49d1a0 关于resize
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite("data/test.jpg",image)
    logger.debug("图片从[%d,%d]被resize成为%r",h,w,image.shape)

    return image,scale

# 看哪个大了，就缩放哪个，规定最大的宽和高：max_width,max_height
def resize_labels(labels,scale):
    if scale==1: return labels
    resized_labels = []
    for label in labels:
        # logger.debug("未缩放bbox label坐标：%r",label)
        _resized_label = [round(x*scale) for x in label]
        resized_labels.append(_resized_label)
        # logger.debug("缩放后bbox label坐标：%r", _resized_label)
    return list(resized_labels)