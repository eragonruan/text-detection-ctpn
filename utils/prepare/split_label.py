import os
import traceback
import cv2 as cv
import numpy as np
from tqdm import tqdm
from utils.prepare.image_utils import orderConvex, shrink_poly
from utils.rpn_msr.config import Config
from utils.prepare import image_utils
'''
    这个程序用来把label，也就是大框
    生成一堆的小框（anchor）
'''
IGNORE_WIDTH = 2 # 小于等于5像素的框，就忽略掉了，太小了，没意义

def split_labels(raw_images_dir, raw_labels_dir, images_dir, labels_split_dir,labels_dir):
    if not os.path.exists(raw_images_dir):
        print("ERROR:图片目录不存在%s" % raw_images_dir)
        exit()
    if not os.path.exists(raw_labels_dir):
        print("ERROR:标签目录不存在%s" % labels_dir)
        exit()

    if not os.path.exists(images_dir): os.makedirs(images_dir)
    if not os.path.exists(labels_split_dir): os.makedirs(labels_split_dir)
    if not os.path.exists(labels_dir): os.makedirs(labels_dir)

    image_names = os.listdir(raw_images_dir)
    image_names.sort()

    for image_name in tqdm(image_names):
        try:
            # 遍历图片目录
            _, fn = os.path.split(image_name)
            label_name, ext = os.path.splitext(fn)
            if ext.lower() not in ['.jpg', '.png']:continue

            # 根据图片目录生成样本文件名
            label_path = os.path.join(raw_labels_dir, label_name + '.txt')
            img_path = os.path.join(raw_images_dir, image_name)

            image = cv.imread(img_path)
            print("读取图片:%s" % img_path)

            resized_image, scale = image_utils.resize_image(image, Config.RPN_IMAGE_WIDTH, Config.RPN_IMAGE_HEIGHT)
            print("调整图像:缩放比例:%f,大小最大宽：%d，最大高：%d" % (scale, Config.RPN_IMAGE_WIDTH, Config.RPN_IMAGE_HEIGHT))

            img_size = resized_image.shape # H,W

            polys = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
            print("读取大框的标签文件：%s，大框标签%d个" % (label_path,len(lines)))

            resized_label_file = open(os.path.join(labels_dir, label_name) + ".txt", "w")
            # 对每一个大框，都完成他的隔16个像素，产生的小框
            for line in lines:
                splitted_line = line.split(',')

                if len(splitted_line)<8: continue
                # print(splitted_line)
                # if len(splitted_line)==0: continue
                x1, y1, x2, y2, x3, y3, x4, y4 =  map(lambda x : scale*x, map(float, splitted_line[:8]))
                resized_label_file.write(",".join(str(x) for x in [x1, y1, x2, y2, x3, y3, x4, y4]))
                resized_label_file.write("\n")
                poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])

                # poly是[4,2]
                poly[:, 0] = poly[:, 0] / img_size[1] * img_size[1] # x / W^2 img-shape=[h,w]
                poly[:, 1] = poly[:, 1] / img_size[0] * img_size[0] # y / H^2

                # 这个是产生一个凸包，并且从左上角排序4个点
                # 这个方法大概读了一下，就是返回闭包的四边形，
                # 结果也不一定是矩形或者平行四边形，感觉是凸四边形
                poly = orderConvex(poly)
                polys.append(poly)

                # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

            resized_label_file.close()
            res_polys = []

            # 对于
            for poly in polys:
                # delete polys with width less than 10 pixel
                # np.linalg.norm(求范数) https://blog.csdn.net/hqh131360239/article/details/79061535
                # 求范数就是求欧氏距离
                if np.linalg.norm(poly[0] - poly[1]) < 5 or np.linalg.norm(poly[3] - poly[0]) < 5:
                    continue

                # 每隔16像素，产生一个小的四边形，返回的是这个4变形的4个坐标
                # 注意，是四边形，不是矩形
                res = shrink_poly(poly)

                # 注意，res是一个[-1,8]的形状，
                res = res.reshape([-1, 4, 2])

                # res reshape后，得到的是3维度了，所以遍历的r就是个2维度的，就是4个点
                # 这个是为了，把4边形，变成矩形，因为GT要求都是矩形的
                # r[x1,y1,x2,y2,x3,y3,x4,y4]=>[x1,y1,x2,y2]
                for r in res:
                    # 4个点里面，的x找一个最小的，当做左上角的x1
                    x_min = np.min(r[:, 0])
                    # 4个点里面，的y找一个最小的，当做左上角的y1
                    y_min = np.min(r[:, 1])
                    # 4个点里面，的x找一个最大的，当做左上角的x2
                    x_max = np.max(r[:, 0])
                    # 4个点里面，的y找一个最大的，当做左上角的y2
                    y_max = np.max(r[:, 1])

                    # 2019.4.7 piginzoo 删除掉首位小于5个像素的点，我觉得没用，删掉，省的干扰，
                    # 只有那个小框的宽度大于5像素，才作为样本GT
                    # 你说会不会漏掉中间的，不会，因为只有大框两边的小框才会距离小于16像素
                    if (x_max-x_min) > IGNORE_WIDTH:
                        res_polys.append([x_min, y_min, x_max, y_max])
                    else:
                        print("小框宽度%d，删除掉" % (x_max-x_min))


            resize_image_path = os.path.join(images_dir,image_name)
            cv.imwrite(resize_image_path, resized_image)
            print("将resized图像保存到：%s" %resize_image_path )

            print("小框标签一个%d个" % len(res_polys))
            with open(os.path.join(labels_split_dir, label_name) + ".txt", "w") as f:
                for p in res_polys:
                    line = ",".join(str(p[i]) for i in range(4))
                    f.writelines(line + "\r\n")

        except Exception as e:
            traceback.print_exc()
            print("Error processing {}".format(image_name))


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir")
    parser.add_argument("--labels_split_dir")
    parser.add_argument("--labels_dir")
    parser.add_argument("--raw_images_dir")
    parser.add_argument("--raw_labels_dir")
    args = parser.parse_args()

    split_labels(
        args.raw_images_dir,
        args.raw_labels_dir,
        args.images_dir,
        args.labels_split_dir,
        args.labels_dir)

