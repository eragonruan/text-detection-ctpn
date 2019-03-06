import os
import sys
import  traceback
import cv2 as cv
import numpy as np
from tqdm import tqdm
from utils import orderConvex, shrink_poly

'''
    这个程序用来把label，也就是大框
    生成一堆的小框（anchor）
'''


sys.path.append(os.getcwd())

DATA_FOLDER = "data/train/"
MAX_LEN = 1200
MIN_LEN = 600

split_labels_path = os.path.join(DATA_FOLDER, "split")
if not os.path.exists(split_labels_path): os.makedirs(split_labels_path)

image_names = os.listdir(os.path.join(DATA_FOLDER, "images"))
image_names.sort()

for image_name in tqdm(image_names):
    try:
        _, fn = os.path.split(image_name)
        label_name, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png']:continue

        label_path = os.path.join(DATA_FOLDER, "labels", label_name + '.txt')
        img_path = os.path.join(DATA_FOLDER, "images", image_name)

        img = cv.imread(img_path)
        img_size = img.shape

        # 不知道原作者为何要设置600和1200的大小限制，我看了他的训练代码，也是没有做啥大小限制的呀，诡异哈
        # 后来明白了，他把原图的训练数据都给resize了，所以，不用管它了
        # im_size_min = np.min(img_size[0:2]) # 找到宽高里小的那个
        # im_size_max = np.max(img_size[0:2]) # 找到宽高里大的那个
        #
        # im_scale = float(600) / float(im_size_min)
        # if np.round(im_scale * im_size_max) > 1200:
        #     im_scale = float(1200) / float(im_size_max)
        # new_h = int(img_size[0] * im_scale)
        # new_w = int(img_size[1] * im_scale)
        #
        # new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        # new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16
        #
        # re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        re_im = img
        re_size = re_im.shape

        polys = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            splitted_line = line.strip().lower().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
            poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
            poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
            poly = orderConvex(poly)
            polys.append(poly)

            # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

        res_polys = []
        for poly in polys:
            # delete polys with width less than 10 pixel
            if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                continue

            res = shrink_poly(poly)
            # for p in res:
               # cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

            res = res.reshape([-1, 4, 2])
            for r in res:
                x_min = np.min(r[:, 0])
                y_min = np.min(r[:, 1])
                x_max = np.max(r[:, 0])
                y_max = np.max(r[:, 1])

                res_polys.append([x_min, y_min, x_max, y_max])

        # cv.imwrite(os.path.join(split_images_path, fn), re_im)
        with open(os.path.join(split_labels_path, label_name) + ".txt", "w") as f:
            for p in res_polys:
                line = ",".join(str(p[i]) for i in range(4))
                f.writelines(line + "\r\n")
                # for p in res_polys:
                #    cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)
                # cv.imshow("demo",re_im)
                # cv.waitKey(0)
    except Exception as e:
        traceback.print_exc()
        print("Error processing {}".format(image_name))
