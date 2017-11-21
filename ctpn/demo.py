from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img,image_name,boxes,scale):
    for box in boxes:
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), (0, 255, 0), 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 255, 0), 2)

    base_name = image_name.split('/')[-1]
    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)

def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))



if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)

