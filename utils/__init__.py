import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def _p_shape(tensor,msg):
    if (FLAGS.debug_mode):
        return tf.Print(tensor, [tf.shape(tensor)], msg,summarize= 100)
    else:
        return tensor

def _p(tensor,msg):
    if (FLAGS.debug_mode):
        return tf.Print(tensor, [tensor], msg,summarize= 100)
    else:
        return tensor

def stat(data):
    return "num={},mean={},std={},max={},min={},<0.5={},<0.7={},0={}".format(
        len(data),
        data.mean(),
        data.std(),
        data.max(),
        data.min(),
        (data < 0.5).sum(),
        (data < 0.7).sum(),
        (data==0).sum())

