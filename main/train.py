import datetime
import os
import sys
import time
import logging
import tensorflow as tf
sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from utils.evaluate.evaluator import *

tf.app.flags.DEFINE_float('learning_rate', 0.01, '') #学习率
tf.app.flags.DEFINE_integer('max_steps', 40000, '') #我靠，人家原来是50000的设置
tf.app.flags.DEFINE_integer('decay_steps', 2000, '')#？？？
tf.app.flags.DEFINE_integer('evaluate_steps',10, '')#？？？
tf.app.flags.DEFINE_float('decay_rate', 0.5, '')#？？？
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')#、、、
tf.app.flags.DEFINE_integer('num_readers', 4, '')#同时启动的进程4个
tf.app.flags.DEFINE_string('gpu', '1', '') #使用第#1个GPU
tf.app.flags.DEFINE_string('model', 'model', '')
tf.app.flags.DEFINE_string('logs_path', 'logs', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')#VGG16的预训练好的模型，这个是直接拿来用的
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_boolean('debug_mode', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

logger = logging.getLogger("Train")
textdetector = TextDetector(DETECT_MODE='H')


def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug_mode):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(FLAGS.logs_path, StyleTime))
    if not os.path.exists(FLAGS.model):
        os.makedirs(FLAGS.model)

    # 输入图像数据的维度[批次,  高度,  宽度,  3通道]
    input_image         = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_image_name    = tf.placeholder(tf.string,  shape=[None,], name='input_image')
    input_bbox          = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox') # 为何是5列？
    input_im_info       = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    adam_opt = tf.train.AdamOptimizer()# 默认是learning_rate是0.001，而且后期会不断的根据梯度调整，一般不用设这个数，所以我索性去掉了

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            # bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
            # cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
            # cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化

            # input_bbox，就是GT，就是样本、标签
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, #预测出来的bbox位移
                                                                                 cls_pred,  #预测出来是否是前景的概率
                                                                                 input_bbox,#标签
                                                                                 input_im_info, # 图像信息
                                                                                 input_image_name)
            # tf.group，是把逻辑上的几个操作定义成一个操作
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            grads = adam_opt.compute_gradients(total_loss)

    apply_gradient_op = adam_opt.apply_gradients(grads, global_step=global_step)

    # 这个是定义召回率、精确度和F1
    v_recall = tf.Variable(0.001, trainable=False)
    v_precision = tf.Variable(0.001, trainable=False)
    v_f1 = tf.Variable(0.001, trainable=False)
    tf.summary.scalar("Recall",v_recall)
    tf.summary.scalar("Precision",v_precision)
    tf.summary.scalar("F1",v_f1)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 某些操作执行的依赖关系，这时我们可以使用tf.control_dependencies()来实现
    # 我依赖于
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op') # no_op啥也不干，但是它依赖的操作都会被干一遍

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.logs_path,StyleTime), tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        logger.info('加载vgg模型：%s',FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
# 上面是定义计算图，下面是真正运行session.run()
#################################################################################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.model)
            logger.debug("最新的模型文件:%s",ckpt) #有点担心learning rate也被恢复
            saver.restore(sess, ckpt)
        else:
            logger.info("从头开始训练模型")
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        # 是的，get_batch返回的是一个generator
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()
        for step in range(FLAGS.max_steps):
            # 注意! 这次返回的只有一张图，以及这张图对应的所有的bbox
            data = next(data_generator) # next(<迭代器>）来返回下一个结果
            logger.debug("在Train中，调用generator从queue中取出一个图片:%r",type(data))
            # data_provider. generator()的返回： yield [im], bbox, im_info # yield很最重要，产生一个generator，可以遍历所有的图片
            # im_info是[w,h,c]

            logger.debug("开始运行sess.run了")
            ml, tl, _, summary_str,bboxs,classes = sess.run([
                                               model_loss,
                                               total_loss,
                                               train_op,
                                               summary_op,
                                               bbox_pred,
                                               cls_prob],
                                              feed_dict={input_image: data[0],
                                                         input_bbox: data[1],
                                                         input_im_info: data[2],
                                                         input_image_name: data[3]}) # data[3]是图像的路径，传入sess是为了调试画图用
            logger.debug("结束运行sess.run了")
            summary_writer.add_summary(summary_str, global_step=step)

            # 修改成为自动的方式，不需要手工调整了，使用了exponential_decay
            # if step != 0 and step % FLAGS.decay_steps == 0:
            #     sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % FLAGS.evaluate_steps == 0:
                avg_time_per_step = (time.time() - start) / FLAGS.evaluate_steps
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step'.format(
                    step, ml, tl, avg_time_per_step))

                # data[4]是大框的坐标，是个数组，8个值
                f1_value,recall_value,precision_value = \
                    generate_big_GT_and_evaluate(bboxs,classes,data[2],data[4])
                sess.run([tf.assign(v_f1, f1_value),
                          tf.assign(v_recall, recall_value),
                          tf.assign(v_precision, precision_value)]
                         )

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.model, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))

# 评估,之前写的，我好像没有评估小框，只评估大框了
# 为什么只评估大框呢？我忘了...
def generate_big_GT_and_evaluate(bboxs,classes,im_info,big_box_labels):
    # 返回所有的base anchor调整后的小框，是矩形
    textsegs, _ = proposal_layer(classes, bboxs, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]  # 这个是小框，是一个矩形

    # 文本检测算法，用于把小框合并成一个4边型（不一定是矩形）, im_info[H,W,C]
    im_info = im_info[0] # 其实就一行，但是为了统一，还是将im_info做成了矩阵
    big_boxes = textdetector.detect(textsegs, scores[:, np.newaxis], (im_info[0],im_info[1]))

    # box是9个值，4个点，8个值了吧，还有个置信度：全部小框得分的均值作为文本行的均值
    big_boxes = np.array(big_boxes, dtype=np.int)
    metrics = evaluate(big_box_labels, big_boxes[:, :8], conf())
    # result = {
    #     'precision': precision,
    #     'recall': recall,
    #     'hmean': hmean,
    #     # 'pairs': pairs,
    # }
    f1_value = metrics['hmean']
    recall_value = metrics['recall']
    precision_value = metrics['precision']
    return f1_value,recall_value,precision_value


if __name__ == '__main__':
    init_logger()
    tf.app.run()
