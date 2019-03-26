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
tf.app.flags.DEFINE_boolean('evaluate_split', True, '') # 是否对小框做出评价
tf.app.flags.DEFINE_string('test_home', 'data/test', '') # 图片主目录
tf.app.flags.DEFINE_string('pred_home', 'data/pred', '') # 图片主目录
tf.app.flags.DEFINE_string('file', '', '')     # 为了支持单独文件
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_boolean('save', True, '')
tf.app.flags.DEFINE_string('model', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS

import logging

logger = logging.getLogger("Train")

RED   =  (0,0,255)
GREEN =  (0,255,0)
GRAY  =  (50,50,50)

# 输入的路径
IMAGE_PATH = "images" # 要文本检测的图片
LABEL_PATH = "labels" # 大框数据，
SPLIT_PATH = "split"  # 小框数据

# 输出的路径
PRED_DRAW_PATH = "draws"   # 画出来的数据
PRED_BBOX_PATH = "detect.bbox" # 探测的小框
PRED_GT_PATH = "detect.gt"     # 探测的大框

# 输入的路径
image_path = os.path.join(FLAGS.test_home, IMAGE_PATH)
label_path = os.path.join(FLAGS.test_home, LABEL_PATH)
split_path = os.path.join(FLAGS.test_home, SPLIT_PATH)

# 输出的路径
pred_draw_path = os.path.join(FLAGS.pred_home, PRED_DRAW_PATH)
pred_gt_path   = os.path.join(FLAGS.pred_home, PRED_GT_PATH)
pred_bbox_path = os.path.join(FLAGS.pred_home, PRED_BBOX_PATH)

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

# 根据图片文件名，得到，对应的标签文件名，可能是split的小框的(矩形4个值)，也可能是4个点的大框的（四边形8个值）
def get_gt_label_by_image_name(image_name,label_path):
    #
    label_name = os.path.splitext(os.path.basename(image_name))  # ['123','png'] 123.png

    if len(label_name)!=2:
        logger.debug("图像文件解析失败：image_name[%s],label_name[%s]", image_name,label_name)
        return None

    label_name = label_name[0]  # /usr/test/123.png => 123
    label_name = os.path.join(label_path, label_name + ".txt")
    if not os.path.exists(label_name):
        logger.debug("标签文件不存在：%s",label_name)
        return None

    bbox = []
    with open(label_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            # logger.debug("line:%s",line)
            line = line.strip().split(",")
            points = list(map(lambda x: int(float(x)), line)) # 用map自动做int转型, float->int是为了防止320.0这样的字符串
            if len(points)>=8: # 处理8个点
                bbox.append(points[:8]) # 去掉最后的一列 置信度
            else:
                bbox.append(points)
    return np.array(bbox)

# 保存预测的输出结果，保存大框和小框，都用这个函数，保存大框的时候不需要scores这个参数
def save(path, file_name,data,scores=None):
    if not FLAGS.save: return

    # 输出
    with open(os.path.join(path, file_name),"w") as f:
        for i, one in enumerate(data):
            line = ",".join([str(value) for value in one])
            if scores is not None:
                line += "," + str(scores[i])
            line += "\r\n"
            f.writelines(line)

# 把框画到图片上
def draw(image,boxes,color,thick=1):
    if not FLAGS.save: return

    if boxes.shape[1]==4: #矩形
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          color=color,
                          thickness=thick)
        return
    if boxes.shape[1]==8: #四边形
        for box in boxes:
            cv2.polylines(image,
                      [box[:8].astype(np.int32).reshape((-1,2))],
                      True,
                      color=color,
                      thickness=thick)
        return

    logger.error("画图失败，无效的Shape:%r",boxes.shape)

def main(argv=None):
    if not os.path.exists(pred_bbox_path): os.makedirs(pred_bbox_path)
    if not os.path.exists(pred_draw_path): os.makedirs(pred_draw_path)
    if not os.path.exists(pred_gt_path): os.makedirs(pred_gt_path)

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

                # 返回所有的base anchor调整后的小框，是矩形
                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5] # 这个是小框，是一个矩形
                logger.debug('textsegs.shape:%r',textsegs.shape)
                logger.debug('score.shape:%r', scores[:, np.newaxis].shape)

                # 做文本检测小框的生成，是根据上面的gt小框合成的
                textdetector = TextDetector(DETECT_MODE='H')

                # 如果关注小框就把小框画上去
                if FLAGS.evaluate_split: draw(img,textsegs,GREEN)

                # 文本检测算法，用于把小框合并成一个4边型（不一定是矩形）
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                # box是9个值，4个点，8个值了吧，还有个置信度：全部小框得分的均值作为文本行的均值
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                logger.info("耗时: %f s" , cost_time)

                # 来！把预测的大框画到图上，输出到draw目录下去，便于可视化观察
                draw(img, boxes[:,:8], color=RED,thick=2)

                # 输出大框到文件
                save(
                    pred_gt_path,
                    os.path.splitext(os.path.basename(im_fn))[0] + ".txt",
                    boxes
                )
                # 输出小框到文件
                save(
                    pred_bbox_path,
                    os.path.splitext(os.path.basename(im_fn))[0] + ".txt",
                    textsegs,
                    scores
                )

                # 对大框作评价
                big_box_labels   = get_gt_label_by_image_name(im_fn,label_path)
                if big_box_labels is not None:
                    logger.debug("找到图像（%s）对应的大框样本（%d）个，开始评测",im_fn,len(big_box_labels))
                    metrics = evaluate(big_box_labels, boxes[:,:8], conf())
                    logger.debug("大框的评价：%r",metrics)
                    logger.debug("将大框标签画到图片上去")
                    draw(img, big_box_labels[:, :8], color=GRAY, thick=2)

                # 对小框做评价
                if FLAGS.evaluate_split:
                    split_box_labels = get_gt_label_by_image_name(im_fn, split_path)
                    if split_box_labels is not None:
                        logger.debug("找到图像（%s）对应的小框split样本（%d）个，开始评测",im_fn, len(split_box_labels))
                        metrics = evaluate(split_box_labels, textsegs, conf())
                        logger.debug("小框的评价：%r", metrics)
                        logger.debug("将小框标签画到图片上去")
                        draw(img, split_box_labels[:,:4], color=GRAY, thick=1)

                out_image_path = os.path.join(pred_draw_path, os.path.basename(im_fn))
                logger.debug("处理后的图像保存到：%s",out_image_path)
                if FLAGS.save: cv2.imwrite(out_image_path, img)#[:, :, ::-1])


if __name__ == '__main__':
    init_logger()
    tf.app.run()
