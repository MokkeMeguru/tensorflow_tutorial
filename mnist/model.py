from enum import Enum, unique
from typing import Callable, List

import tensorflow as tf
from attrdict import AttrDict
from tensorflow.contrib.framework import add_arg_scope


@unique
class ImageValType(str, Enum):
    RGB_UNSIGNED = 'unsigned'
    RGB_SIGNED = 'signed'
    RGB_ABSNORMALISED = 'absnormalized'
    RGB_NORMALISED = 'normalized'


@unique
class Optimizer(str, Enum):
    ADAM = 'adam'
    SGD = 'sgd'


@unique
class Activation(str, Enum):
    RELU = "relu"


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
    if hps.model_params.activation == Activation.RELU:
        activation = tf.nn.relu
    else:
        raise NotImplementedError(
                'activation is in {}'.format(list(Activation)))
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
