import numpy as np
import tensorflow as tf
from nets import vgg
from tensorflow.contrib import slim

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(image, bbox, im_info):
    image = mean_image_subtraction(image)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)

    return conv5_3




def loss(predictions, labels):
    model_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('model loss', model_loss)
    tf.summary.scalar('total loss', total_loss)
    return model_loss,total_loss