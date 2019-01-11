import tensorflow as tf
from tensorflow.contrib import slim

from nets import vgg


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C])
        net.set_shape([None, None, input_channel])

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def model(image, bbox, im_info):
    image = mean_image_subtraction(image)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)

    rpn_conv = slim.conv2d(conv5_3, 512, 3)

    lstm_output = Bilstm(rpn_conv, 512, 128, 512, scope_name='BiLSTM')

    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")

    return bbox_pred, cls_pred


def loss():
    return 1.0
