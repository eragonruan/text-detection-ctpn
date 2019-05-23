import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import logging

def _p_shape(tensor,msg):
    if (FLAGS.debug):
        return tf.Print(tensor, [tf.shape(tensor)], msg,summarize= 100)
    else:
        return tensor

def _p(tensor,msg):
    if (FLAGS.debug):
        return tf.Print(tensor, [tensor], msg,summarize= 100)
    else:
        return tensor

def stat(data):
    if len(data)==0: return "data size is 0"
    return "num={},mean={},std={},max={},min={},<0.5={},<0.7={},0={}".format(
        len(data),
        data.mean(),
        data.std(),
        data.max(),
        data.min(),
        (data < 0.5).sum(),
        (data < 0.7).sum(),
        (data==0).sum())



def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])
