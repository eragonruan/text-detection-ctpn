import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import  test_ctpn
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
from text_proposal_connector import TextProposalConnector

CLASSES = ('__background__',
           'text')


def connect_proposal(text_proposals, scores, im_size):
    cp = TextProposalConnector()
    line = cp.get_text_lines(text_proposals, scores, im_size)
    return line

def save_results(image_name,im,line,thresh):
    inds=np.where(line[:,-1]>=thresh)[0]
    if len(inds)==0:
        return 

    for i in inds:
        bbox=line[i,:4]
        score=line[i,-1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(0,0,255),thickness=1)
    image_name=image_name.split('/')[-1]
    cv2.imwrite(os.path.join("data/results",image_name),im)


def check_img(im):
    im_size = im.shape
    if max(im_size[0:2]) < 600:
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        start_row = int((600 - im_size[0]) / 2)
        start_col = int((600 - im_size[1]) / 2)
        end_row = start_row + im_size[0]
        end_col = start_col + im_size[1]
        img[start_row:end_row, start_col:end_col, :] = im
        return img
    else:
        return im


def ctpn(sess, net, image_name):
    img = cv2.imread(image_name)
    im = check_img(img)
    timer = Timer()
    timer.tic()
    scores, boxes = test_ctpn(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    
    keep = np.where(dets[:, 4] >= 0.7)[0]
    dets = dets[keep, :]
    line = connect_proposal(dets[:, 0:4], dets[:, 4], im.shape)
    save_results(image_name, im, line,thresh=0.9)


if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print ('Loading network {:s}... '.format("VGGnet_test")),
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(os.getcwd(),"checkpoints/model_final.ckpt"))
    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {:s}'.format(im_name))
        ctpn(sess, net, im_name)
