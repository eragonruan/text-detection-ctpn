from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file

if __name__ == "__main__":
    cfg_from_file('ctpn/text.yml')

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    net = get_network("VGGnet_test")
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    print(' done.')

    print('all nodes are:\n')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    node_names = [node.name for node in input_graph_def.node]
    for x in node_names:
        print(x)
    output_node_names = 'Reshape_2,rpn_bbox_pred/Reshape_1'
    output_graph_def = convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))
    output_graph = 'data/ctpn.pb'
    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()
