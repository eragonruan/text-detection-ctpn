# encoding:utf-8
import os
import time
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("data provider")

from utils.dataset.data_util import GeneratorEnqueuer

DATA_FOLDER = "data/train/"

# 扎到image目录下所有的图片文件，返回的是一个文件列表
def get_training_data():
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "images")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


# 这个很重要，去寻找这张图片对应的标注
# https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/
#
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1192, 1862, 2424, 1895
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line) # 用map自动做int转型
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox # 返回四个坐标的数组

# 装载大框
def load_big_GT(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        points = list(map(lambda x: int(float(x)), line))  # 用map自动做int转型, float->int是为了防止320.0这样的字符串
        bbox.append(points[:8])  # 去掉最后的一列 置信度
    return bbox # 返回四个坐标的数组


def generator(vis=False):
    image_list = np.array(get_training_data())
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])

    while True:
        np.random.shuffle(index)
        for i in index: # 遍历所有的图片文件
            try:
                im_fn = image_list[i] # fn file name，文件名
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im_info = np.array([h, w, c]).reshape([1, 3]) # 变成1行，3列，在这个是shape啊，不是数据

                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)

                # 这个很重要，去寻找这张图片对应的标注
                # https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/
                #
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # 1192, 1862, 2424, 1895
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                #
                #
                txt_fn = os.path.join(DATA_FOLDER, "split", fn + '.txt')
                big_gt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')

                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                if not os.path.exists(big_gt_fn):
                    print("大框 Big Ground truth for image {} not exist!".format(big_gt_fn))
                    continue
                bbox = load_annoataion(txt_fn)
                big_gt = load_big_GT(big_gt_fn)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue
                if len(big_gt) == 0:
                    print("Big Ground truth for image {} empty!".format(im_fn))
                    continue

                if vis: # 给丫画出来
                    for p in bbox:
                        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
                    axs.imshow(im[:, :, ::-1])
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                logger.debug("generator yield了一个它读出的图片[%s]")
                # 卧槽，注意看，这次返回的只有一张图
                yield [im], bbox, im_info,big_gt # yield很最重要，产生一个generator，可以遍历所有的图片

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    try:
        # 这里又藏着一个generator，注意，这个函数get_batch()本身就是一个generator
        # 但是，这里，他的肚子里，还藏着一个generator()
        # 这个generator实际上就是真正去读一张图片，返回回来了
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    logger.debug("从GeneratorEnqueuer的queue中取出的图片")
                    break
                else:
                    time.sleep(0.01)
            # yield一调用，就挂起，等着外面再来调用next()了
            # 所以，可以看出来queue.get()出来的是一个图片，验证了我的想法，就是一张图，不是多张
            yield generator_output

            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, vis=True)
    while True:
        image, bbox, im_info = next(gen)
        print('done')
