import datetime
import os
import sys
import time
import tensorflow as tf
sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model

tf.app.flags.DEFINE_integer('batch_size', 8, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_boolean('restore', True, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 5000, '')
tf.app.flags.DEFINE_integer('decay_steps', 40000, '')
tf.app.flags.DEFINE_integer('decay_rate', 0.1, '')
FLAGS = tf.app.flags.FLAGS


def tower_loss(images, score_maps, vertex_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_vertex, f_geometry = model.model(images, is_training=True)

    model_loss, classification_loss, vertex_loss, localization_loss = \
        model.loss(score_maps, f_score, vertex_maps, f_vertex, geo_maps, f_geometry, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, classification_loss, vertex_loss, localization_loss

def main(argv=None):
    gpus = range(len(FLAGS.gpu_list.split(',')))
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
  
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)


    tower_grads = []
    reuse_variables = None
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
               
                total_loss, model_loss, classification_loss, vertex_loss, localization_loss = \
                    tower_loss(iis, isms, ivms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

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
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size * len(gpus))

        start = time.time()
        for step in range(restore_step, FLAGS.max_steps):
            data = data_generator.next()

            ml, tl, cl, vl, ll, _, summary_str = sess.run([model_loss, total_loss, classification_loss,
                                                           vertex_loss, localization_loss, train_op, summary_op],
                                                          feed_dict={input_images: data[0], input_score_maps: data[2],
                                                                     input_vertex_maps: data[3],
                                                                     input_geo_maps: data[4],
                                                                     input_training_masks: data[5]})

            summary_writer.add_summary(summary_str, global_step=step)

            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                # avg_examples_per_second = (10 * FLAGS.batch_size * len(gpus)) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, cls loss {:.4f}, vertex loss {:.4f}, loc loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                        step, ml, tl, cl, vl, ll, avg_time_per_step, learning_rate.eval()))

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('m_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))


if __name__ == '__main__':
    tf.app.run()
