import tensorflow as tf
import tf_slim as slim

from nets import vgg
from utils.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def make_var(name, shape, initializer=None):
    return tf.compat.v1.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.compat.v1.variable_scope(scope_name) as scope:
        shape = tf.shape(input=net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C])
        net.set_shape([None, None, input_channel])

        lstm_fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.compat.v1.keras.initializers.VarianceScaling(scale=0.01, mode=('FAN_AVG').lower(), distribution=("uniform" if False else "truncated_normal"))
        init_biases = tf.compat.v1.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    with tf.compat.v1.variable_scope(scope_name) as scope:
        shape = tf.shape(input=net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])

        init_weights = tf.compat.v1.keras.initializers.VarianceScaling(scale=0.01, mode=('FAN_AVG').lower(), distribution=("uniform" if False else "truncated_normal"))
        init_biases = tf.compat.v1.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def model(image):
    image = mean_image_subtraction(image)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)

    rpn_conv = slim.conv2d(conv5_3, 512, 3)

    lstm_output = Bilstm(rpn_conv, 512, 128, 512, scope_name='BiLSTM')

    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(input=cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(input=cls_pred_reshape)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
                          name="cls_prob")

    return bbox_pred, cls_pred, cls_prob


def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    with tf.compat.v1.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.compat.v1.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info, [16, ], [16]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(value=tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(value=rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(value=rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(value=rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    with tf.compat.v1.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(bbox_pred, cls_pred, bbox, im_info):
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(input=cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])
    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.compat.v1.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
    rpn_label = tf.gather(rpn_label, rpn_keep)
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

    # box loss
    rpn_bbox_pred = bbox_pred
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]

    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

    rpn_loss_box_n = tf.reduce_sum(input_tensor=rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])

    rpn_loss_box = tf.reduce_sum(input_tensor=rpn_loss_box_n) / (tf.reduce_sum(input_tensor=tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(input_tensor=rpn_cross_entropy_n)

    model_loss = rpn_cross_entropy + rpn_loss_box

    regularization_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.compat.v1.summary.scalar('model_loss', model_loss)
    tf.compat.v1.summary.scalar('total_loss', total_loss)
    tf.compat.v1.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.compat.v1.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
