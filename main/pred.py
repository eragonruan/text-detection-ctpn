# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

from utils.evaluate.evaluator import *

tf.app.flags.DEFINE_boolean('debug_mode', True, '')
tf.app.flags.DEFINE_string('home', 'data/test', '') # 图片主目录
tf.app.flags.DEFINE_string('file', '', '')     # 为了支持单独文件
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('model', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS

import logging

logger = logging.getLogger("Train")

IMAGE_PATH = "images" # 要文本检测的图片
DRAW_PATH = "draws"   # 画出来的数据
LABEL_PATH = "labels" # 大框数据，
# SPLIT_PATH = "split"  # 小框数据

image_path = os.path.join(FLAGS.home, IMAGE_PATH)
draw_path =  os.path.join(FLAGS.home, DRAW_PATH)
label_path = os.path.join(FLAGS.home, LABEL_PATH)

def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug_mode):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])


def get_images():
    if FLAGS.file != "":
        return [os.path.join(image_path, FLAGS.file)]

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for img_name in os.listdir(image_path):
        for ext in exts:
            if img_name.endswith(ext):
                files.append(os.path.join(image_path, img_name))
                break
    logger.debug('找到需要检测的图片%d张',len(files))
    return files

def get_gt_label(label_file):
    bbox = []
    with open(label_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        # logger.debug("line:%s",line)
        line = line.strip().split(",")
        four_points = list(map(int, line)) # 用map自动做int转型
        bbox.append(four_points)
    return bbox # 返回四个坐标的数组


def main(argv=None):
    if not os.path.exists(image_path): os.makedirs(image_path)
    if not os.path.exists(draw_path): os.makedirs(draw_path)
    if not os.path.exists(label_path): os.makedirs(label_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)
        # bbox_pred  ( N , H , W , 40 )
        # cls_pred   ( N , H , W*10 , 2 )
        # cls_prob  ( N , H , W*10 , 2 ), 但是，二分类，对是、不是，又做了一个归一化
        # 看一下，这个是对每一个FeatureMap得到的，比如是16x50，然后800点每一个又衍生出10个anchor
        # 然后需要对每个anchor来处理了

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.model)
            logger.debug("从路径[%s]查找到最新的checkpoint文件[%s]",FLAGS.model,ckpt_state)
            model_path = os.path.join(FLAGS.model, os.path.basename(ckpt_state.model_checkpoint_path))
            logger.info('从%s加载模型',format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                logger.info("正在探测图片：%s",im_fn)
                start = time.time()
                try:
                    img = cv2.imread(im_fn)#[:, :, ::-1] # bgr是opencv通道默认顺序
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                # 之前的代码是resize，归一化一下，我觉得没有必要，不resize了
                # img, (rh, rw) = resize_image(im
                # img = im
                # (rh,rw) = im.shape

                h, w, c = img.shape
                logger.debug('图像的h,w,c:%d,%d,%d',h,w,c)
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                # 返回所有的图片的小的anchor对应的box
                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')

                logger.debug('textsegs.shape:%r',textsegs.shape)
                logger.debug('score.shape:%r', scores[:, np.newaxis].shape)

                for one_text_seg in textsegs:
                    cv2.rectangle(img,
                              (one_text_seg[0],one_text_seg[1]),
                              (one_text_seg[2],one_text_seg[3]),
                              color=(0, 255,0),
                              thickness=1)

                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                logger.info("耗时: %f s" , cost_time)

                for i, box in enumerate(boxes):
                    # logger.debug("画框:%r到图像[%s]",[box[:8].astype(np.int32).reshape((-1, 1, 2))],im_fn)

                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                                   thickness=2)
                    cv2.putText(img, str(i),(box[0],box[1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(0, 0, 255),
                                lineType=1)

                # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                out_image_path = os.path.join(draw_path, os.path.basename(im_fn))
                logger.debug("处理后的图像保存到：%s",out_image_path)
                cv2.imwrite(out_image_path, img)#[:, :, ::-1])

                with open(os.path.join(draw_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)

                # 作评价
                label_name = os.path.splitext(os.path.basename(im_fn))
                if len(label_name)==2:
                    label_name = label_name[0] # /usr/test/123.png => 123
                    label_name = os.path.join(label_path,label_name+".txt")
                    if os.path.exists(label_name):
                        logger.info("存在GT标签文件[%s]，进行评价：" , label_name)
                        gt_labels = get_gt_label(label_name)
                        methods, metrics = \
                        evaluate_method(gt_labels, textsegs, conf())
                        logger.info(methods)
                        logger.info(metrics)

if __name__ == '__main__':
    init_logger()
    tf.app.run()
