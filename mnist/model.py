from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from attrdict import AttrDict
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.examples.tutorials import mnist
from tqdm import tqdm

import params
import params_utils


def load_mnist(
        save_mnist_path: Path = Path('./tmp/mnist'), show_info: bool = True):
    """load mnist dataset via tensorflow

    Parameters
    ----------
    save_mnist_path: Path
    show_info: bool

    Returns
    -------
    dataset: tf.dataset
        tf.dataset

    Examples
    -----
    >>> dataset = load_mnist()
    >>> len(dataset.train.images), len(dataset.validation.images)
    55000, 5000
    >>> len(dataset.test.images)
    10000
    >>> np.shape(dataset.train.image[0])
    784 (= 28 * 28)
    """
    dataset = mnist.input_data.read_data_sets(str(save_mnist_path.absolute()),
                                              one_hot=True)
    if show_info:
        print(dataset)
        train = dataset.train
        sample_images = train.images[0:16]
        sample_labels = train.labels[0:16]
        fig = plt.figure(figsize=(4, 4))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(16):
            img = np.reshape(sample_images[i], (28, 28))
            fig.add_subplot(4, 4, i + 1)
            plt.title(np.argmax(sample_labels[i], axis=0))
            plt.tick_params(bottom=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
            plt.imshow(img, cmap='gray_r')
        plt.show()
    return dataset


@add_arg_scope
def convnet(scope_name: str,
            x: tf.Tensor,
            filters: int,
            kernel_size: List[int] = [5, 5],
            pool_size: List[int] = [2, 2],
            activation: Callable = tf.nn.relu):
    with tf.name_scope(scope_name):
        conv = tf.layers.conv2d(
            x, filters=filters, kernel_size=kernel_size, padding='SAME')
        activation = tf.nn.relu(conv)
        pooling = tf.layers.max_pooling2d(
            activation, pool_size=pool_size, strides=2)
    return pooling


def model_fn(x: tf.Tensor,
             hps: AttrDict = None,
             is_training: tf.Tensor = None):
    """mnist classification model with cnn

    Args
    ----
    x: tf.Tensor
        input tensor [B, H, W, C]
    hps: AttrDict
        model parameters
    is_training: tf.Tensor[None]
        train -> tf.Variable(True)
        inference or validate -> tf.Variable(False)

    Returns
    -------
    x: tf.Tensor
        output tensor [B, NUM_CLASS]
    """
    if hps.model_params.activation == params_utils.Activation.RELU:
        activation = tf.nn.relu
    else:
        raise NotImplementedError(
                'activation is in {}'.format(list(params_utils.Activation)))
    x = convnet('convnet1', x,
                filters=hps.model_params.conv_filters[0],
                kernel_size=hps.model_params.conv_kernels[0],
                pool_size=hps.model_params.pooling_size[0],
                activation=activation)
    x = convnet('convnet2', x,
                filters=hps.model_params.conv_filters[1],
                kernel_size=hps.model_params.conv_kernels[1],
                pool_size=hps.model_params.pooling_size[1],
                activation=activation)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,
                        units=hps.model_params.mid_dense_units,
                        activation=activation)
    x = tf.layers.dropout(
        x, rate=hps.model_params.dropout_rate,
        training=tf.cond(is_training, lambda: True, lambda: False))
    x = tf.layers.dense(x,
                        units=hps.data_params.num_class,
                        activation=tf.nn.softmax)
    return x


def train(path_prefix: str = 'tmp'):
    # import parmeter data
    hps = params_utils.parse_params(params.mnist_classification_args)

    # load dataset
    dataset = load_mnist(show_info=False)

    # inputs setting
    with tf.variable_scope('input'):
        if hps.data_params.is_flattened:
            shape = [None, np.prod(hps.data_params.image_size)]
        else:
            shape = [None] + list(hps.data_params.image_size)
        x = tf.placeholder(tf.float32, shape)
        y = tf.placeholder(tf.float32, [None, 10])
        is_training = tf.placeholder(tf.bool, shape=None)
    _x = tf.reshape(x, [-1, 28, 28, 1])
        # if hps.data_params.is_flattened:
        #     if len(hps.data_params.image_size) == 3:
        #         x = tf.reshape(x, [-1] + list(hps.data_params.image_size))
        #     elif len(hps.data_params.image_size) == 2:
        #         x = tf.reshape(
        #             x, [-1] + list(hps.data_params.image_size) + [1])
        #     else:
        #         raise NotImplementedError('image shape is NHW or NHWC')
    y_hat = model_fn(_x, hps, is_training)

    # with tf.variable_scope('input'):
    #     if hps.data_params.is_flattened:
    #         shape = [None, np.prod(hps.data_params.image_size)]
    #     else:
    #         shape = [None] + list(hps.data_params.image_size)
    #     x = tf.placeholder(tf.float32, [None, 784])
    
    #     print('shape', shape)
    #     # if hps.data_params.is_flattened:
    #     #     if len(hps.data_params.image_size) == 3:
    #     #         x = tf.reshape(x, [-1] + list(hps.data_params.image_size))
    #     #     elif len(hps.data_params.image_size) == 2:
    #     #         x = tf.reshape(
    #     #             x, [-1] + list(hps.data_params.image_size) + [1])
    #     #     else:
    #     #         raise NotImplementedError('image shape is NHW or NHWC')
    #     is_training = tf.placeholder(tf.bool, shape=None)
    #     y = tf.placeholder(tf.float32, [None, 10])
    # x = tf.reshape(x, [256, 28, 28, 1])
    # # input -> model -> output
    # y_hat = model_fn(x, hps=hps, is_training=is_training)

    # setup metrics
    with tf.name_scope('metrics'):
        with tf.name_scope('accuracy'):
            correctness = tf.equal(tf.argmax(y_hat), tf.argmax(y))
            correctness = tf.cast(correctness, tf.float32)
            accuracy = tf.reduce_mean(correctness)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y, logits=y_hat)
            loss = tf.reduce_mean(cross_entropy)

        # note:
        # xxx is THE value, xxx_op is the OPERATION to update xxx
        with tf.name_scope('train'):
            train_loss, train_loss_op = tf.metrics.mean(
                loss, name='train_loss')
            train_acc, train_acc_op = tf.metrics.mean(
                accuracy, name='train_acc')
            tf.summary.scalar('train/loss', train_loss, collections=['train'])
            tf.summary.scalar('train/acc', train_acc, collections=['train'])

        with tf.name_scope('val'):
            val_loss, val_loss_op = tf.metrics.mean(loss, name='val_loss')
            val_acc, val_acc_op = tf.metrics.mean(accuracy, name='val_acc')
            tf.summary.scalar('val/loss', val_loss, collections=['val'])
            tf.summary.scalar('val/acc', val_acc, collections=['val'])

        # metrics initializer
        train_metrics_initialzie_op = tf.variables_initializer(
            [var for var in tf.local_variables()
             if 'train/' in var.name])
        val_metrics_initialize_op = tf.variables_initializer(
            [var for var in tf.local_variables()
             if 'val/' in var.name])

        # gathered summary operation
        train_summary_op = tf.summary.merge_all('train')
        val_summary_op = tf.summary.merge_all('val')

    # optimizer settings
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, False)
        learning_rate = hps.hyper_parameters.learning_rate
        if hps.hyper_parameters.optimizer == params_utils.Optimizer.ADAM:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
        elif hps.hypter_paramters.optimizer == params_utils.Optimizer.SGD:
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
        else:
            raise NotImplementedError(
                    'optimizer is in {}'.format(list(params_utils.Optimizer)))
        train_step = optimizer.minimize(loss, global_step=global_step)

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    saver = tf.train.Saver()
    save_path = hps.paths.model_path / path_prefix
    ckpt = tf.train.get_checkpoint_state(save_path)

    with tf.Session() as sess:
        if ckpt:
            print('restore variable')
            last_model = ckpt.model_checkpoint_path
            saver.restore(sess, last_model)
            sess.run(train_metrics_initialzie_op)
            sess.run(val_metrics_initialize_op)
            writer = tf.summary.FileWriter(
                hps.paths.log_path / path_prefix, None)

        else:
            # initialize all variable and operations
            sess.run(init_op)
            sess.run(local_init_op)
            sess.run(train_metrics_initialzie_op)
            sess.run(val_metrics_initialize_op)
            sess.run(init_op)
            writer = tf.summary.FileWriter(
                hps.paths.log_path / path_prefix, sess.graph)

        for step in tqdm(range(hps.hyper_parameters.step)):
            batch = dataset.train.next_batch(hps.hyper_parameters.batch_size)
            sess.run(train_step,
                     feed_dict={x: batch[0], y: batch[1], is_training: True})
            # train_log
            if step % 100:
                summary, gstep = sess.run(
                    [train_summary_op, global_step])
                writer.add_summary(summary, global_step=gstep)
                sess.run(train_metrics_initialzie_op)
                saver.save(sess, save_path + '/model.ckpt',
                           global_step=global_step)
            # validation log
            if step % 1000:
                sess.run(val_metrics_initialize_op)
                for _ in range(50):
                    val_batch = dataset.train.next_batch(100)
                    sess.run([val_loss_op, val_acc_op],
                             feed_dict={x: val_batch[0], y: val_batch[1],
                                        is_training: False})
                summary, gstep = sess.run([val_summary_op, global_step])
                writer.add_summary(summary, global_step=gstep)


if __name__ == '__main__':
    train()
