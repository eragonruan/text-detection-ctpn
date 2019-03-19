import numpy as np
from shapely.geometry import Polygon

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

    # 取整一下吧
    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    p = x_min
    res.append([p, int(k1 * p + b1), # kx+b, p相当于是x，第一个p是最左面，但是第二个p就是16位取整的值了
                start - 1, int(k1 * (p + 15) + b1),
                start - 1, int(k2 * (p + 15) + b2),
                p, int(k2 * p + b2)])

    for p in range(start, end + 1, r):
        res.append([p, int(k1 * p + b1),
                    (p + 15), int(k1 * (p + 15) + b1),
                    (p + 15), int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
    return np.array(res, dtype=np.int).reshape([-1, 8])
