from pathlib import Path

import numpy as np
import tensorflow as tf


def simple_model(x: tf.Tensor, scope: str = 'simple_model'):
    weights_scope = scope + '/weights'
    with tf.variable_scope(weights_scope):
        a = tf.get_variable(
            name=weights_scope + '_a',
            shape=[], dtype=tf.float32,
            initializer=tf.initializers.constant(
                value=0.0, dtype=tf.float32),
            trainable=True)
        b = tf.get_variable(
            name=weights_scope + '_b',
            shape=[], dtype=tf.float32,
            initializer=tf.initializers.constant(
                value=0.0, dtype=tf.float32),
            trainable=True)
    with tf.name_scope(scope + '/formula'):
        y = a * x + b
    with tf.name_scope(scope + '/log'):
        log_op = tf.strings.format('a - {} / b - {}', [a, b])
    return y, log_op


def teacher(x: np.float32):
    return 5.0 * x + 8.0


def train(args):
    with tf.variable_scope('inputs'):
        x = tf.placeholder(dtype=tf.float32, shape=[])
        y = tf.placeholder(dtype=tf.float32, shape=[])

    # setup model
    y_hat, log_op = simple_model(x)
    loss_op = tf.math.squared_difference(y, y_hat)

    #  setup tensorboard log
    path = Path('./tmp/sgd_func')
    tf.summary.scalar('loss', loss_op)
    summary_op = tf.summary.merge_all()

    # setup stdout log
    logs_op = tf.print(
        tf.strings.join(
            [log_op, tf.strings.format('loss - {}', loss_op)], '/'
        ))
    # setup hyper parameter
    learning_rate = 1e-3

    # setup optimizer
    global_step = tf.Variable(0, False, name='global_step')
    optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
    train_step = optimizer.minimize(loss_op, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # save model
    saver = tf.train.Saver()
    save_path = "./tmp/sgd_save"
    ckpt = tf.train.get_checkpoint_state(save_path)

    with tf.Session() as sess:
        if ckpt:
            print('restore variable')
            last_model = ckpt.model_checkpoint_path
            saver.restore(sess, last_model)
            writer = tf.summary.FileWriter(path, None)

        else:
            sess.run(init_op)
            writer = tf.summary.FileWriter(path, sess.graph)

        for step in range(500):
            _x = np.random.uniform()
            _y = teacher(_x)
            feed_dict = {x: _x, y: _y}
            if step % 100 == 0:
                summary, _, _=sess.run(
                    [summary_op, train_step, logs_op], feed_dict)
                saver.save(sess, save_path + "/model.ckpt",
                           global_step=global_step)
            else:
                summary, _, = sess.run(
                    [summary_op, train_step], feed_dict)
            writer.add_summary(summary, step)

def parse():
    """Parse Args
    note:
    in ipython, it don't use argparse
    """
    return None

def main():
    args = parse()
    train(args)

if __name__ == '__main__':
    main()
