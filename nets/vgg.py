import tensorflow as tf
import logging

logger = logging.getLogger("vgg")

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

# 把图片扔给VGG16，
def vgg_16(inputs, scope='vgg_16'):
    logger.debug("输入数据shape=(%r)",inputs.get_shape())
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            #这里没说，其实核是[3,3,3]，多出来的3是颜色3通道
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            #出来的是m x n x 64维度的图像
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # 出来的是m/2 x n/2 x 64维度的图像，
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # 出来的是m/2 x n/2 x 128维度的图像，核其实是[64,3,3]，64是上层的feature map的深度
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            #出来的是m/4 x n/4 x 128维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # 出来的是m/4 x n/4 x 256维度的图像，核其实是[128,3,3]
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # 出来的是m/8 x n/8 x 256维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # 出来的是m/8 x n/8 x 512维度的图像，核其实是[256,3,3]
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # 出来的是m/16 x n/16 x 512维度的图像
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # 出来的是m/16 x n/16 x 512维度的图像
            # 细节，最后这个单元，并没有再继续到1024个核，而是还是继续用512个核
            #最终，出来的图像是 （m/16 x n/16 x 512）
            logger.debug("输入数据shape=(%r)", net.get_shape())

    return net
