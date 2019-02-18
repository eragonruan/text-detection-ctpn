import tensorflow as tf
from tensorflow.contrib import slim

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
    return tf.get_variable(name, shape, initializer=initializer)

#net：h/16 x w/16 x 512
#               512,           128,             512
def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        #N批次， H现在是h/16, W是w/16， C是512（h,w是原图大小）
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C])# 为何这么reshape，就是：我LSTM每个时间点，输入是512维度的向量，输入W个时间片，但是我按照 N*H做为批次灌
        #这里暗含着的意思就是说，我不在关心行与行之间的关系了，我只关心这一行的，其他行都被我混淆成批次的概念了
        #可是，这个时候的一行是什么呢？别忘了，是被池化和卷积后的，所以，对应着原图就是一大片啊，至于多少，我有点算不清楚了，3x3池化应该是N+2把
        #至少是16倍，如果只算池化，但是还有卷积呢，卷积会多出周边的点，但是不会放大缩小（说的是视野），
        #我理解是((1*2+2)*2+2)*2+2)*2+2)=46的像素把，另外，池化做几次，最后又做了池化1次，姑且认为是50个像素把

        #reshape完不就是这样么，干嘛又set_shape一下，不解？？？
        net.set_shape([None, None, input_channel])

        #128个隐含单元的前向LSTM 1个
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        # 128个隐含单元的后向LSTM 1个
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1) #双向的出来的结果是隐含层128的维度，concat就是256维度了

        #所以出来的LSTM的维度就变成了这个，256维度的，
        #这里需要澄清一下，应该是一次都出来，这个一次包含两个概念，一个是RNN的时间片个数，一个是之前的批次(N*H)
        #所以一起输出出来，应该是 （N*H, W, 256)的，
        #但是这里给变成了2维度的，这个细节要稍微注意一下，不知道目的，继续往后看
        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        #[NHW，256] x [256,512] => [NHW,512]
        #干呢呢？这尼玛就是一个全连接啊
        #保持清醒：这个时候，原始图像的维度，从3维度(RGB)，变成了512，一行已经乱掉了，不过可以通过reshape恢复
        outputs = tf.matmul(lstm_out, weights) + biases

        #上面刚说完，果然，这里又给恢复回去了，变成了N，H，W，512，
        #具体是多少呢：N, h/16, w/16, 512了，N是批次
        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs

#[N,h/16,w/16,512] 512            40或20 40:bbox_pred, 20:cls_pred
#这个函数在干嘛？他是一个全链接网络，输出rpn_bbox_pred（文字框大小），或者是rpn_cls_score(分类概率，是否包含文字）
def lstm_fc(net,   input_channel, output_channel, scope_name):
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])#又给reshape了，变态啊

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        #[N*(h/16)*(w/16),512] * [512,40/20] = [ N*H*W, 40],最后变成40维度，或者20个维度的东西了
        #为是40，或者20呢？
        #40：bbox，那就是10的anchor的y和h（y坐标和高度）
        #这篇文章：https://zhuanlan.zhihu.com/p/34757009 说的都应该是20，我还可以理解
        #可是，代码里面的bbox，居然是40，诡异？继续看吧，看后面到底有啥用途？
        output = tf.matmul(net, weights) + biases#<-----这就是一个全连接网络啊

        #输出[ N*H*W, 40/20],然后再分开 [ N,H,W,40/20]
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def model(image):
    image = mean_image_subtraction(image)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)

    rpn_conv = slim.conv2d(conv5_3, 512, 3)#最后的3，其实就是[3,3]的核定义，卧槽，又做了一个512卷积，干，

    #理清思路，rpn_conv现在还是 h/16 x w/16 x 512的张量
    #Bilstm(net,                   input_channel, hidden_unit_num, output_channel, scope_name):
    lstm_output = Bilstm(rpn_conv, 512,           128,             512, scope_name='BiLSTM')

    #好，理清一下思路，现在lstm_output是啥呢？
    #是 N x h/16 x w/16 x 512的向量，
    #但是这512，已经浓缩了关于顺序的抽象语义在里面了，另外，这一个点可是表示至少50个像素的"感受野"

    #lstm_fc(net,                    input_channel, output_channel, scope_name):
    #LSTM出来的东西，灌到一个全连接FC网络里，得出回归的框的坐标
    bbox_pred = lstm_fc(lstm_output, 512,           10 * 4, scope_name="bbox_pred")
    #输出的bbox_pred [N,H,W,40]，bbox_pred

    # LSTM出来的东西，灌到一个全连接FC网络里，得出是否包含文字的概率
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")
    #输出的cls_pred [N,H,W,20]，20个是10个anchor的对于是/不是置信概率

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)

    #卧槽，reshape成哪样啊，这是要？
    #懂了，就是拆成每个anchor的，[N,H,W,20] => [N,H,W*10,2]
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    #cls_pred_reshape_shape=( N , H , W*10 , 2 )

    cls_prob = tf.reshape(
        #干嘛呢？把10个anchor的概率值，做了归一化，比如说，10个框里面包含东西的概率是xxx,然后
        #就又做了一个归一化，把这10个都包含的概率在做了一个归一化的概率分布出来，
        tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
        [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
        name="cls_prob")#然后再给reshape回来，折腾啊！

    return bbox_pred, cls_pred, cls_prob#<----记住，这厮是归一化的
    #bbox_pred  ( N , H , W , 40 )
    #cls_pred   ( N , H , W*10 , 2 )
    # cls_prob  ( N , H , W*10 , 2 ), 但是，二分类，对是、不是，又做了一个归一化


def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info, [16, ], [16]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]

#Smooth_L1：
def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

#损失函数
# bbox_pred,
# cls_pred,
# bbox,
# im_info
def loss(bbox_pred, cls_pred, bbox, im_info):
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])
    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
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

    rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

    model_loss = rpn_cross_entropy + rpn_loss_box

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
