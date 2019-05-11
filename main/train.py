import datetime
import os
import time
import tensorflow as tf
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider
from utils.text_connector.detectors import TextDetector
from utils.evaluate.evaluator import *
from main import pred
from main.early_stop import  EarlyStop

tf.app.flags.DEFINE_string('name', 'ctpn', '')
tf.app.flags.DEFINE_float('learning_rate', 0.01, '') #学习率
tf.app.flags.DEFINE_integer('max_steps', 40000, '') #我靠，人家原来是50000的设置
tf.app.flags.DEFINE_integer('decay_steps', 2000, '')#？？？
tf.app.flags.DEFINE_integer('evaluate_steps',10, '')#？？？
tf.app.flags.DEFINE_float('decay_rate', 0.5, '')    #？？？
tf.app.flags.DEFINE_float('max_lr_decay', 3, '')    #？？？
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('train_images_dir','','')
tf.app.flags.DEFINE_string('train_labels_dir','','')
tf.app.flags.DEFINE_string('train_labels_split_dir','','')
tf.app.flags.DEFINE_string('validate_images_dir','','')
tf.app.flags.DEFINE_string('validate_labels_dir','','')
tf.app.flags.DEFINE_string('validate_labels_split_dir','','')
tf.app.flags.DEFINE_integer('validate_batch',30,'')
tf.app.flags.DEFINE_integer('early_stop',5,'')
tf.app.flags.DEFINE_integer('num_readers', 4, '')#同时启动的进程4个
tf.app.flags.DEFINE_string('gpu', '1', '') #使用第#1个GPU
tf.app.flags.DEFINE_string('model', 'model', '')
tf.app.flags.DEFINE_float('lambda1', 1000, '')
tf.app.flags.DEFINE_string('logs_path', 'logs/tboard', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')#VGG16的预训练好的模型，这个是直接拿来用的
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_boolean('debug', False, '')
tf.app.flags.DEFINE_boolean('resize', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

logger = logging.getLogger("Train")
textdetector = TextDetector(DETECT_MODE='H')


def init_logger():
    level = logging.DEBUG
    if(FLAGS.debug):
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=level,
        handlers=[logging.StreamHandler()])

def main(argv=None):

    # 选择GPU
    if FLAGS.gpu!="1" and FLAGS.gpu!="0":
        logger.error("无法确定使用哪一个GPU，退出")
        exit()
    logger.info("使用GPU%s显卡进行训练",FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger.info(
        "本次使用的参数：\nlearning_rate:%f\ndecay_steps:%f\nmax_steps:%d\nevaluate_steps:%d\nmodel:%s\nlambda1:%d\nlogs_path:%s\nrestore:%r\ndebug:%r\nsave_checkpoint_steps:%d", \
        FLAGS.learning_rate,
        FLAGS.decay_steps,
        FLAGS.max_steps,
        FLAGS.evaluate_steps,
        FLAGS.model,
        FLAGS.lambda1,
        FLAGS.logs_path,
        FLAGS.restore,
        FLAGS.debug,
        FLAGS.save_checkpoint_steps)

    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(FLAGS.logs_path, StyleTime))
    if not os.path.exists(FLAGS.model):
        os.makedirs(FLAGS.model)

    logger.info("CTPN训练开始...")

    # 输入图像数据的维度[批次,  高度,  宽度,  3通道]
    input_image         = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_image_name    = tf.placeholder(tf.string,  shape=[None,], name='input_image_name')
    input_bbox          = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox') # 为何是5列？
    input_im_info       = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    adam_opt = tf.train.AdamOptimizer(learning_rate)# 默认是learning_rate是0.001，而且后期会不断的根据梯度调整，一般不用设这个数，所以我索性去掉了

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            # bbox_pred  ( N , H , W , 40 )                N:批次  H=h/16  W=w/16 ，其中 h原图高    w原图宽
            # cls_pred   ( N , H , W*10 , 2 )              每个(featureMap H*W个)点的10个anchor的2分类值，（所以是H*W*10*2个）
            # cls_prob  ( N , H , W*10 , 2 ), 但是，对是、不是，又做了一个归一化

            # input_bbox，就是GT，就是样本、标签
            total_loss, model_loss, \
            rpn_cross_entropy, rpn_loss_box = \
                model.loss(bbox_pred, #预测出来的bbox位移
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
    # 早停用的变量
    early_stop = EarlyStop(FLAGS.early_stop,FLAGS.max_lr_decay)


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
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers,
                                                 data_dir=FLAGS.train_images_dir,
                                                 label_dir=FLAGS.train_labels_dir,
                                                 label_split_dir=FLAGS.train_labels_split_dir)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        average_train_time = 0

        for step in range(FLAGS.max_steps):
            train_start = time.time()

            # 注意! 这次返回的只有一张图，以及这张图对应的所有的bbox
            data = next(data_generator) # next(<迭代器>）来返回下一个结果
            logger.debug("在Train中，调用generator从queue中取出一个图片:%r",type(data))
            # data_provider. generator()的返回： yield [im], bbox, im_info # yield很最重要，产生一个generator，可以遍历所有的图片
            # im_info是[w,h,c]

            image = data[0][0]
            bbox_label = data[1]
            scale = 1

            logger.info("开始第%d步训练，运行sess.run",step)
            image = image[:,:,::-1]
            ml, tl, _, summary_str,classes = sess.run([
                                               model_loss,
                                               total_loss,
                                               train_op,
                                               summary_op,
                                               cls_prob],
                                              feed_dict={input_image: [image],
                                                         input_bbox: bbox_label,
                                                         input_im_info: np.array(image.shape).reshape([1, 3]),
                                                         input_image_name: data[3]}) # data[3]是图像的路径，传入sess是为了调试画图用
            average_train_time = average_time(train_start,average_train_time,step)
            logger.info("结束第%d步训练，结束sess.run，平均每个step时间：%f,modeloss:%f,totaloss:%f",step,average_train_time,ml,tl)
            summary_writer.add_summary(summary_str, global_step=step)

            if step!=0 and step % FLAGS.evaluate_steps == 0:
                validate_start = time.time()
                logger.info("在第%d步，开始进行模型评估",step)
                # data[4]是大框的坐标，是个数组，8个值
                f1_value,recall_value,precision_value = \
                    validate(sess,bbox_pred, cls_prob, input_im_info, input_image)
                # 更新F1,Recall和Precision
                sess.run([tf.assign(v_f1, f1_value),
                          tf.assign(v_recall, recall_value),
                          tf.assign(v_precision, precision_value)])
                logger.info("在第%d步，模型评估结束，耗时：%f，f1=%f,recall=%f,precision=%f", step, time.time() - validate_start,f1_value,recall_value,precision_value)

                if is_need_early_stop(early_stop,f1_value,saver,sess,step,learning_rate,train_start_time): break


def average_time(start_time,avg_time,step):
    _time = time.time() - start_time
    if avg_time == 0:
        avg_time = _time
    else:
        # 移动平均
        avg_time = avg_time - avg_time / step + _time / step
    return avg_time


def is_need_early_stop(early_stop,f1_value,saver,sess,step,learning_rate,train_start_time):
    decision = early_stop.decide(f1_value)

    if decision == EarlyStop.ZERO: # 当前F1是0，啥也甭说了，继续训练
        return False

    if decision == EarlyStop.CONTINUE:
        logger.info("新F1值比最好的要小，继续训练...")
        return False

    if decision == EarlyStop.BEST:
        logger.info("新F1值[%f]大于过去最好的F1值，早停计数器重置，并保存模型", f1_value)
        save_model(saver, sess, step, train_start_time)
        return False

    if decision == EarlyStop.STOP:
        logger.warning("超过早停最大次数，也尝试了多次学习率Decay，无法在提高：第%d次，训练提前结束", step)
        return True

    if decision == EarlyStop.LEARNING_RATE_DECAY:
        logger.info("学习率(learning rate)衰减：%f=>%f", learning_rate.eval(), learning_rate.eval() * FLAGS.decay_rate)
        sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))
        return False

    logger.error("无法识别的EarlyStop结果：%r",decision)
    return True

def save_model(saver, sess, step, train_start_time):
    # 每次训练的模型不要覆盖，前缀是训练启动时间
    filename = ('ctpn-{:s}-{:d}'.format(train_start_time, step + 1) + '.ckpt')
    filename = os.path.join(FLAGS.model, filename)
    saver.save(sess, filename)
    logger.info("在第%d步，保存了模型文件(checkout point)：%s", step, filename)


# 用来批量验证
# 入参： t_xxxx，都是张量
def validate(sess,
             t_bbox_pred, t_cls_prob, t_input_im_info, t_input_image):

    #### 加载验证数据,随机加载FLAGS.validate_batch张
    image_list, image_names = data_provider.get_validate_images_data(FLAGS.validate_images_dir,
                                                                     FLAGS.validate_batch)

    precision_sum=recall_sum=f1_sum = 0
    for i in range(len(image_list)):

        image = image_list[i]
        image_name = image_names[i]

        # session, t_bbox_pred, t_cls_prob, t_input_im_info, t_input_image, d_img
        boxes, scores, textsegs = pred.predict_by_network(sess,t_bbox_pred, t_cls_prob, t_input_im_info, t_input_image,image)

        bbox_pred = boxes[:,:8]

        # 得到标签名字
        GT_labels = pred.get_gt_label_by_image_name(image_name,FLAGS.validate_labels_dir)
        metrics = evaluate(GT_labels,bbox_pred,conf())
        precision_sum += metrics['precision']
        recall_sum += metrics['recall']
        f1_sum += metrics['hmean']
        logger.debug("图片%s的探测结果的精确度:%f,召回率:%f,F1:%f",image_name,
                     metrics['precision'],metrics['recall'],metrics['hmean'])
    precision_mean = 0
    recall_mean = 0
    f1_mean =  0
    if len(image_list)>0:
        precision_mean = precision_sum / len(image_list)
        recall_mean = recall_sum / len(image_list)
        f1_mean = f1_sum / len(image_list)
        
    logger.debug("这批%d个图片的平均的精确度:%f,召回率:%f,F1:%f",
                 len(image_list),precision_mean,recall_mean,f1_mean)

    return precision_mean,recall_mean,f1_mean

if __name__ == '__main__':
    init_logger()
    tf.app.run()
