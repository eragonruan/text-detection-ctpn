# coding=utf-8
import math
import os
import shutil
import sys
from functools import cmp_to_key
import cv2
import numpy as np
import tensorflow as tf
from nets import model as model


tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_rctw/', '')
tf.app.flags.DEFINE_integer('output_mode', 8, '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, size=32, max_side_len=2400):
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % size == 0 else (resize_h // size) * size
    resize_w = resize_w if resize_w % size == 0 else (resize_w // size) * size

    if resize_w < 32:
        resize_w = 32
    if resize_h < 32:
        resize_h = 32

    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def main(argv=None):
    import os
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_vertex, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            timer = Timer()

            total_time = 0.0
            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image!")
                    continue

                #scale = [0.3, 1.0]
                #im_list, h_list, w_list = multi_resize_image(im, max_side_len=2400, scale=scale)

                im_list, h_list, w_list = multi_resize_by_long_side(im,32,[512])
                boxes_list = []
                timer.tic()
                for i, img in enumerate(im_list):
                    score, vertex, geometry = sess.run([f_score, f_vertex, f_geometry], feed_dict={input_images: [img]})
                    boxes = post_line(img, score, vertex, geometry, score_map_thresh=0.8, box_thresh=0.12)

                    # 恢复到原图大小
                    if len(boxes):
                        bb = boxes[:, :8].reshape((-1, 4, 2))
                        bb[:, :, 0] /= w_list[i]
                        bb[:, :, 1] /= h_list[i]
                        bb = bb.reshape((-1, 8))
                        boxes[:, :8] = bb

                    boxes_list.extend(boxes)

                boxes = np.array(boxes_list, dtype=np.float).reshape([-1, 9])
                boxes = standard_nms(boxes, 0.2)

                # 存结果
                if len(boxes):
                    with open(FLAGS.output_path + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]),
                              'w') as f:
                        for box in boxes:
                            score = box[8]
                            box = box[:8].reshape([4, 2])
                            box = orderConvex(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                                continue
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(0, 255, 0), thickness=2)

                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), im[:, :, ::-1])

            print("totl time: {}".format(total_time))


if __name__ == '__main__':
    tf.app.run()
