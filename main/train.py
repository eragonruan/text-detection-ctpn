import datetime
import os
import sys
import time

import tensorflow as tf

sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider

# tf.app.flags.DEFINE_float('learning_rate', 1e-5, '') #学习率
tf.app.flags.DEFINE_float('learning_rate', 0.1, '') #学习率
tf.app.flags.DEFINE_integer('max_steps', 20000, '') #我靠，人家原来是50000的设置
tf.app.flags.DEFINE_integer('decay_steps', 4000, '')#？？？
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')#？？？
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')#、、、
tf.app.flags.DEFINE_integer('num_readers', 4, '')#同时启动的进程4个
tf.app.flags.DEFINE_string('gpu', '1', '') #使用第#1个GPU
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')#VGG16的预训练好的模型，这个是直接拿来用的
tf.app.flags.DEFINE_boolean('restore', True, '')
tf.app.flags.DEFINE_boolean('debug_mode', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 200, '')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

import logging

logger = logging.getLogger("Train")

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
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    # 输入图像数据的维度[批次,  高度,  宽度,  3通道]
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

    # ？？？
    input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox') # 为何是5列？
    # ？？？
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    learning_rate = tf.train.exponential_decay(
                                        FLAGS.learning_rate, # 初始化的learning rate
                                        global_step,         # 全局步数计数器，我理解是不管epochs多少，不停的把每个epochs内的step累加
                                        FLAGS.decay_steps, # 决定衰减周期，就是隔这么多step就开始衰减一下
                                        FLAGS.decay_rate,  # 每次衰减的倍率，就是变成之前的多少
                                        staircase = True)


    tf.summary.scalar('learning_rate', learning_rate)
    adam_opt = tf.train.AdamOptimizer(learning_rate)

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            # bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
            # cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
            # cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化

            # input_bbox，就是GT，就是样本、标签
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                 input_im_info)
            # tf.group，是把逻辑上的几个操作定义成一个操作
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            grads = adam_opt.compute_gradients(total_loss)

    apply_gradient_op = adam_opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 某些操作执行的依赖关系，这时我们可以使用tf.control_dependencies()来实现
    # 我依赖于
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op') # no_op啥也不干，但是它依赖的操作都会被干一遍

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            logger.debug("最新的日志文件:%s",ckpt)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            saver.restore(sess, ckpt)
            logger.info("从之前的checkpoint继续训练，步数是: {}".format(restore_step))
        else:
            logger.info("从头开始训练模型")
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        # 是的，get_batch返回的是一个generator
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()
        for step in range(restore_step, FLAGS.max_steps):
            # 注意! 这次返回的只有一张图，以及这张图对应的所有的bbox
            data = next(data_generator) # next(<迭代器>）来返回下一个结果
            logger.debug("在Train中，调用generator从queue中取出一个图片:%r",type(data))
            # data_provider. generator()的返回： yield [im], bbox, im_info # yield很最重要，产生一个generator，可以遍历所有的图片
            # im_info是[w,h,c]

            logger.debug("开始运行sess.run了")
            ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                              feed_dict={input_image: data[0],
                                                         input_bbox: data[1],
                                                         input_im_info: data[2]})
            logger.debug("结束运行sess.run了")
            summary_writer.add_summary(summary_str, global_step=step)

            # 在干什么？Adam的learning rate是自动衰减的呀，这里为何要再调整lr？！
            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ml, tl, avg_time_per_step, learning_rate.eval()))

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))


if __name__ == '__main__':
    init_logger()
    tf.app.run()
