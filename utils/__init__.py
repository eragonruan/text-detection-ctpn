import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def _p_shape(tensor,msg):
    if (FLAGS.debug_mode):
        return tf.Print(tensor, [tf.shape(tensor)], msg,summarize= 100)
    else:
        return tensor