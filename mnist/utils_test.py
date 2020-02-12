import tensorflow as tf

import utils


def conv2d_straight(name='v1'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
    x = utils.conv2d(scope_name=name + '_1', x=x, n_out=3)
    x = utils.conv2d(scope_name=name + '_2', x=x, n_out=3)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.summary.FileWriter('./test/conv2d_straight', sess.graph)


def conv2d_reuse(name='v2'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
    x = utils.conv2d(scope_name=name + '_1', x=x, n_out=3)
    x = utils.conv2d(scope_name=name + '_1', x=x, n_out=3, reuse=True)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.summary.FileWriter('./test/conv2d_reuse', sess.graph)

def linear_single(name='v1'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 128])
    x = utils.linear(scope_name=name + '_1', x=x, n_out=128)
    x = utils.linear(scope_name=name + '_2', x=x, n_out=128)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.summary.FileWriter('./test/linear_single', sess.graph)


def linear(name='v2'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
    x = utils.linear(scope_name=name + '_1', x=x, n_out=3)
    x = utils.linear(scope_name=name + '_2', x=x, n_out=3)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.summary.FileWriter('./test/linear', sess.graph)


def linear_reuse(name='v3'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
    x = utils.linear(scope_name=name+'_1', x=x, n_out=3)
    x = utils.linear(scope_name=name+'_1', x=x, n_out=3, reuse=True)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.summary.FileWriter('./test/linear_reuse', sess.graph)


def main():
    conv2d_straight()
    tf.reset_default_graph()
    conv2d_reuse()
    tf.reset_default_graph()
    linear()
    tf.reset_default_graph()
    linear_reuse()
    tf.reset_default_graph()
    linear_single()


if __name__ == '__main__':
    main()
