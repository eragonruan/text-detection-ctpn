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
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
from text_proposal_connector import TextProposalConnector

CLASSES = ('__background__',
           'text')


def connect_proposal(text_proposals, scores, im_size):
    cp = TextProposalConnector()
    line = cp.get_text_lines(text_proposals, scores, im_size)
    return line

def save_results(image_name,im,im_scale,line,thresh):
    inds=np.where(line[:,-1]>=thresh)[0]
    image_name=image_name.split('/')[-1]
    if len(inds)==0:
        im = cv2.resize(im, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join("data/results",image_name),im)
        return 

    for i in inds:
        bbox=line[i,:4]
        score=line[i,-1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(0,255,0),thickness=2)
    im = cv2.resize(im, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results",image_name),im)


def check_img(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return re_im, im_scale


def ctpn(sess, net, image_name):
    img = cv2.imread(image_name)
    im, im_scale = check_img(img)
    timer = Timer()
    timer.tic()
    scores, boxes = test_ctpn(sess, net, im)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    
    keep = np.where(dets[:, 4] >= 0.7)[0]
    dets = dets[keep, :]
    line = connect_proposal(dets[:, 0:4], dets[:, 4], im.shape)
    save_results(image_name, im,im_scale, line,thresh=0.9)


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
        #ckpt=tf.train.get_checkpoint_state("output/ctpn_end2end/voc_2007_trainval/")
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    print (' done.')

    #saver.restore(sess, os.path.join(os.getcwd(),"checkpoints/model_final_tf13.ckpt"))
    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
