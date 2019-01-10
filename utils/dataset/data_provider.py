# encoding:utf-8
import time
import os, random
import numpy as np
import matplotlib.pyplot as plt
from utils.data_util import GeneratorEnqueuer
import cv2

DATA_FOLDER = "/home/slade/code/generate/SynthText/data/croped/"
charset_path = 'data/charset_synthtext.txt'

SPACE_INDEX = 0
SPACE_TOKEN = ''
TARGET_HEIGHT = 32
TARGET_WIDTH = 256

def get_encode_decode_maps():
    char = ''
    with open(charset_path) as f:
        for ch in f.readlines():
            ch = ch.strip('\r\n')
            char = char + ch

    char = 'ず'+char
    print('nclass:', len(char))

    id_to_char = {i: j for i, j in enumerate(char)}
    char_to_id = {j: i for i, j in enumerate(char)}
    return id_to_char,char_to_id


def get_training_data():
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(DATA_FOLDER):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_charset(charset_path):
    charset = dict()
    with open(charset_path) as f:
        lines = f.readlines()
        for cnt,line in enumerate(lines):
            line = line.strip()
            charset[line] = cnt+1
    return charset

def load_label(img_path,charset):
    txt_path = img_path[:-3] + 'txt'
    with open(txt_path) as f:
        line = f.readline()
        label = []
        for i in range(len(line)):
            label.append(charset[line[i]])
    return label


def ReadImg(img_path,charset):
    img = cv2.imread(img_path)
    labels = load_label(img_path,charset)
    return img, labels


def ProcessImg(im, label):
    # rgb2gray
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # resize and pad image
    h, w, _ = im.shape
    ratio = float(TARGET_HEIGHT) / h

    resize_h = TARGET_HEIGHT
    resize_w = int(w * ratio)

    if resize_w >= TARGET_WIDTH:
        im = cv2.resize(im, (int(TARGET_WIDTH), int(TARGET_HEIGHT)))
    else:
        img = cv2.resize(im, (int(resize_w), int(resize_h)))
        # padding to taget width
        im = np.zeros([TARGET_HEIGHT, TARGET_WIDTH, 3])
        im[:, :resize_w,:] = img
    im = np.array(im).reshape([TARGET_HEIGHT,TARGET_WIDTH,3])
    label_vec = [int(c) for c in label]
    label_vec = np.array(label_vec).reshape([-1])

    return im, label_vec



def generator(batch_size=1):
    
                images.append(im)
                labels.extend(label_vec)
                label_lens.extend(label_len)
                time_steps.extend(time_step)

                if len(label_lens)==batch_size:
                    yield images, labels, label_lens, time_steps
                    images = []
                    labels = []
                    label_lens = []
                    time_steps=[]

            except Exception as e:
                print(e)
                # import traceback
                # traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, batch_size=2, vis=False)
    while True:
        images, labels, label_lens, time_steps = next(gen)
        print('done')
