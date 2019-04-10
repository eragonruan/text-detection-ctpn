import tensorflow as tf
from tensorflow.contrib import slim
import logging
from nets import vgg
from utils.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from utils import _p_shape,_p
FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger('model_train')

# [123.68, 116.78, 103.94] 这个是VGG的预处理要求的，必须减去这个均值：https://blog.csdn.net/smilejiasmile/article/details/80807050
def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1] # 通道数
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    # 干啥呢？ 按通道，多分出一个维度么？
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
        # 每个通道感觉都减去了一个数，貌似标准化一样
        # 不过这个 means 是如何决定的呢？

    return tf.concat(axis=3, values=channels) # 然后把图片再合并回来


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)

#net：h/16 x w/16 x 512
#               512,           128,             512
def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):

    net = _p_shape(net, "LSTM输入")

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
        # Bi-LSTM的输出是： [batch_size, max_time, cell_fw.output_size]
        # 对应到我们这里就是：[N*H,        W       , 512  ],恰好输出还是512
        # 这么理解的话，LSTM输出是所有的时间片上的都输出，而不是最后一个，这是我想搞明白的


        # 所以出来的LSTM的维度就变成了这个，256维度的，
        # 这里需要澄清一下，应该是一次都出来，这个一次包含两个概念，
        # 一个是RNN的时间片个数，一个是之前的批次(N*H)
        # 所以一起输出出来，应该是 （N*H, W, 256)的，
        # 但是这里给变成了2维度的，这个细节要稍微注意一下，不知道目的，继续往后看
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

        outputs = _p_shape(outputs,"LSTM输出")

        return outputs

#[N,h/16,w/16,512] 512            40或20 40:bbox_pred, 20:cls_pred
#这个函数在干嘛？他是一个全链接网络，输出rpn_bbox_pred（文字框大小），或者是rpn_cls_score(分类概率，是否包含文字）
def lstm_fc(net,   input_channel, output_channel, scope_name):
    net =  _p_shape(net, "LSTM后的FC输入")
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])#又给reshape了，变态啊
        #注意一个细节，维度只是C，也就是512，其他的都变成批次了

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)

        #weight有是一个矩阵[input_channel x output_channel]，
        #对于bbox预测：input_channel=512, output_channel=10 * 4
        #对于cls分类预测：input_channel=512, output_channel=10 * 2
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        # [N*(h/16)*(w/16),512] * [512,40/20] = [ N*H*W, 40],最后变成40维度，或者20个维度的东西了
        # 为什么是40，或者20呢？
        # 40：bbox，那就是10的anchor的y和h（y坐标和高度）
        # 这篇文章：https://zhuanlan.zhihu.com/p/34757009 说的都应该是20，我还可以理解
        # 可是，代码里面的bbox，居然是40，诡异？继续看吧，看后面到底有啥用途？
        output = tf.matmul(net, weights) + biases#<-----这就是一个全连接网络啊

        #输出[ N*H*W, 40/20],然后再分开 [ N,H,W,40/20]
        output = tf.reshape(output, [N, H, W, output_channel])

    return _p_shape(output, "LSTM后的FC输出")


def model(image):
    image = _p_shape(image, "最开始输入")

    image = mean_image_subtraction(image)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)

        conv5_3 = _p_shape(conv5_3, "VGG的5-3卷基层输出")

    # 再做 512个卷积核[3x3]
    rpn_conv = slim.conv2d(conv5_3, 512, 3)#最后的3，其实就是[3,3]的核定义，卧槽，又做了一个512卷积，干，
    logger.debug("rpn_conv的维度:%r",rpn_conv.get_shape())

    # 理清思路，rpn_conv现在还是 h/16 x w/16 x 512的张量
    # Bilstm(net,                   input_channel, hidden_unit_num, output_channel, scope_name):
    lstm_output = Bilstm(rpn_conv, 512,            128,             512, scope_name='BiLSTM')
    logger.debug("lstm_output输出的维度:%r", lstm_output.get_shape())

    # 好，理清一下思路，现在lstm_output是啥呢？
    # 是 N x h/16 x w/16 x 512的向量，
    # 但是这512，已经浓缩了关于顺序的抽象语义在里面了，另外，这一个点可是表示至少50个像素的"感受野"

    #  lstm_fc(net,                    input_channel, output_channel, scope_name):
    #  LSTM出来的东西，灌到一个全连接FC网络里，得出回归的框的坐标
    bbox_pred = lstm_fc(lstm_output, 512,           10 * 4, scope_name="bbox_pred")
    # 输出的bbox_pred [N,H,W,40]，bbox_pred
    # 为何全链接，也就是bbox_pred输出是 10 * 4？？？
    logger.debug("lstm之后又做了一个FC(bbox_pred)，输出的维度:%r", bbox_pred.get_shape())

    #  LSTM出来的东西，灌到一个全连接FC网络里，得出是否包含文字的概率
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")
    # 输出的cls_pred [N,H,W,20]，20个是10个anchor的对于是/不是置信概率
    # 未做softmax归一化之前的隐含层输出
    logger.debug("lstm之后另一分支也做了一个FC(cls_pred)，输出的维度:%r", cls_pred.get_shape())

    #  transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)

    # 卧槽，reshape成哪样啊，这是要？
    # 懂了，就是拆成每个anchor的，[N,H,W,20] => [N,H,W*10,2]
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    # cls_pred_reshape_shape=( N , H , W*10 , 2 )

    cls_prob = tf.reshape(
        # 干嘛呢？把10个anchor的概率值，做了归一化，比如说，10个框里面包含东西的概率是xxx,然后
        # 就又做了一个归一化，把这10个都包含的概率在做了一个归一化的概率分布出来，
        tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
        [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
        name="cls_prob")#然后再给reshape回来，折腾啊！

    # return bbox_pred, cls_pred, cls_prob#<----记住，这厮是归一化的
    return _p_shape(bbox_pred, "bbox_pred"), \
            _p_shape(cls_pred, "cls_pred"), \
            _p_shape(cls_prob, "cls_prob")
    # bbox_pred  ( N , H , W , 40 )
    # cls_pred   ( N , H , W*10 , 2 )
    # cls_prob  ( N , H , W*10 , 2 ), 但是，二分类，对是、不是，又做了一个归一化


# bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
# cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
# cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化
#
# _anchor_target_layer 主要功能是计算获得属于rpn网络的label。
# https://zhuanlan.zhihu.com/p/32230004
# 通过对所有的anchor与所有的GT计算IOU，
# 由此得到 rpn_labels, rpn_bbox_targets,
# rpn_bbox_inside_weights, rpn_bbox_outside_weights
# 这4个比较重要的第一次目标label，通过消除在图像外部的 anchor，
# 计算IOU >=0.7 为正样本，IOU <0.3为负样本，
# 得到在理想情况下应该各自一半的256个正负样本
# （实际上正样本大多只有10-100个之间，相对负样本偏少）。
def anchor_target_layer(cls_pred, bbox, im_info, input_image_name, scope_name):
    with tf.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        # tf.py_func是把普通函数改造成TF运行用的函数：包装一个普通的 Python 函数，这个函数接受 numpy 的数组作为输入和输出，
        # 让这个函数可以作为 TensorFlow 计算图上的计算节点 OP 来使用。
        # https://zhuanlan.zhihu.com/p/32970370
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred,bbox,im_info,[16, ],[16],input_image_name],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')
        # rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare，是不是包含前景

        rpn_labels = _p_shape(rpn_labels,"rpn_labels tensor")
        rpn_bbox_targets  = _p_shape(rpn_bbox_targets,"rpn_bbox_targets tensor")
        rpn_bbox_inside_weights = _p_shape(rpn_bbox_inside_weights,"rpn_bbox_inside_weights tensor")
        rpn_bbox_outside_weights = _p_shape(rpn_bbox_outside_weights,"rpn_bbox_outside_weights tensor")

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
# bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
# cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
# cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化
def loss(bbox_pred, cls_pred, bbox, im_info,input_image_name):

    bbox_pred = _p_shape(bbox_pred,"Loss输入：bbox_pred")

    #返回 [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]
    #rpn_labels anchor是否包含前景
    #rpn_bbox_targets 所有的anchor对应的4个标签回归值，所有对应在图像内的anchors
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info,input_image_name, "anchor_target_layer")

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2]) #2是指两个概率值，(1, H, WxA, 2)
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2]) #(HxWxA, d)
    rpn_label = tf.reshape(rpn_data[0], [-1]) #rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare，是不是包含前景
    # ignore_label(-1)

    ################################################################################
    # 看！这里很重要，去掉了所有的-1，也就是刨除了那些不用的样本，只保留了300个样本 *************
    ################################################################################
    fg_keep = tf.where(tf.equal(rpn_label,1)) # 所有前景，fg_keep是一个true/false数组,tf.equal返回true/false test结果的数组
    # 这步很重要，因为anchor_target_layer_py那个函数里面只标注了batch/2个正样本，和batch/2个负样本，剩下的anchor都标注成-1了，所以，这里要去掉他们
    # 最后，只剩下batch个了
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1)) # rpn只剩下是非-1的那些元素的下标，注意！是位置下标！

    # tf.gather 类似于数组的索引，可以把向量中某些索引值提取出来，得到新的向量，适用于要提取的索引为不连续的情况。这个函数似乎只适合在一维的情况下使用。
    # https://blog.csdn.net/Cyiano/article/details/76087747
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep) # 把对应的前景的概率取出来，rpn_cls_score是从cls_pred来的(具体自己读代码)
    rpn_label = tf.gather(rpn_label, rpn_keep) # 看，rpn_label是所有的label，但是经过rpn_keep过滤，就只剩下batch的样本了

    logger.debug("我们来看看lstm预测的cls_pred和anchor_target_layer选出来的anchor组成的label的shape：")

    # 做交叉熵，1那个是通过IoU算出来的，而rpn_cls_score是通过卷积网络算出来的
    # loss11111111111111111
    rpn_label = _p_shape(rpn_label, "做交叉熵:label")
    rpn_label = _p(rpn_label,"做交叉熵:label")
    rpn_cls_score = _p_shape(rpn_cls_score, "做交叉熵:predict")
    rpn_cls_score = _p(rpn_cls_score, "做交叉熵:predict")

    # 文档里说，输入的logits应该是网络的输出，而不需要做softmax
    #  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
    # on `logits` internally for efficiency.  Do not call this op with the
    # output of `softmax`, as it will produce incorrect results.
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

    # box loss
    rpn_bbox_pred = bbox_pred
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]

    # 看！这里又过滤了一下 ====> "rpn_keep"，只保留样本
    # # https://github.com/eragonruan/text-detection-ctpn/issues/334
    # 我理解，应该是rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), fg_keep)  # shape (N, 4)
    # 因为，计算bbox的loss的时候，应该只考虑前景框的loss，对于背景，也就是负样本（我理解负样本就是那些非前景的anchor们），这些负样本计算他们的loss没有任何意义。#
    # 不知道我理解的对不对？如果是的话，这应该是一个bug。
    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), fg_keep)  # shape (N, 4) #256个，好像是，anchor_target_layer完成了采样，不过这块需要回头再看看
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), fg_keep)
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), fg_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), fg_keep)

    rpn_bbox_pred = _p_shape(rpn_bbox_pred,"我们来看看lstm预测的bbox_pred和anchor_target_layer选出来的anchor组成的bbox_targets的shape：")
    rpn_bbox_pred = _p_shape(rpn_bbox_pred,"rpn_bbox_pred")
    rpn_bbox_targets = _p_shape(rpn_bbox_targets, "rpn_bbox_targets")

    # loss2222222222222222222，用的叫smooth l1，说防止梯度爆炸之类的，
    # <https://zhuanlan.zhihu.com/p/32230004>
    # "论文提到的 _smooth_l1_loss 相当于一个二次方函数和直线函数的结合，但是为什么要这样呢？不太懂，论文说它比较鲁棒，没有rcnn中使用的L2 loss 那么对异常值敏感，当回归目标不受控制时候，使用L2 loss 会需要更加细心的调整学习率以避免梯度爆炸？_smooth_l1_loss消除了这个敏感性。"
    # 关于smooth L1：
    # https://zhuanlan.zhihu.com/p/48426076
    # L2对异常点敏感，他里面局那个例子，真实值为1，1000的异常值就会影响巨大，L1因为没有平方，所以这个异常值1000影响就小一些，
    # smooth L1和L1-loss函数的区别在于，L1-loss在0点处导数不唯一，可能影响收敛。smooth L1的解决办法是在0点附近使用平方函数使得它更加平滑。
    # smooth L1 loss让loss对于离群点更加鲁棒，即：相比于L2损失函数，其对离群点、异常值（outlier）不敏感，梯度变化相对更小，训练时不容易跑飞。
    rpn_loss_box_n = tf.reduce_sum(
        rpn_bbox_outside_weights * smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)),
            reduction_indices=[1])

    # 是对loss求个均值，总loss / 前景的anchhor的个数（这个前景是配置里面规定的批次的一半的数量）
    # 表示，平均每个前景anchor的loss
    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
    # lambda1是照着论文来的，来平衡分类和回归，需要根据实际情况调整，让两者在一个数量级
    model_loss = rpn_cross_entropy + FLAGS.lambda1 * rpn_loss_box

    # tf.get_collection：从一个集合中取出全部变量，是一个列表
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
