from pathlib import Path
from typing import Any, Callable, List, Union

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.examples.tutorials import mnist


@add_arg_scope
def linear(scope_name: str,
           x: tf.Tensor,
           n_out: int,
           bias: bool=True,
           weight_norm: bool=True,
           initializer: Union[str, Callable[[Any], tf.Tensor]]
           = tf.initializers.glorot_normal(),
           reuse: bool = False):
    """Linear for any Tensor

    Parameters
    ----------
    scope_name: str
        scope name
    x: tf.Tensor
        input tensor [B, *, n_in]
    n_out: int
        output tensor's last dimention
    bias: bool
        use bias
    weight_norm: bool
        apply weight normalization to weight tensor
    initializer
        initializer for weight tensor
    reuse: bool
        reuse variable

    Returns
    -------
    x: tf.Tensor
        output tensor [B, * , n_out]
    """
    rank = len(x.get_shape())
    with tf.variable_scope('linear_' + scope_name, reuse=reuse):
        n_in = int(x.get_shape()[-1])
        w = tf.get_variable("W", [n_in, n_out],
                            tf.float32, initializer=initializer)
        b = tf.get_variable('b', [n_out],
                            tf.float32, initializer=tf.initializers.zeros)
    with tf.name_scope('linear'):
        if weight_norm:
            w = tf.nn.l2_normalize(w, [0])
        if rank > 2:
            x = tf.tensordot(x, w, [[rank-1], [0]])
        else:
            x = tf.matmul(x, w)
        x = tf.add(x, b)
    return x


@add_arg_scope
def conv2d(scope_name: str,
           x: tf.Tensor,
           n_out: int,
           channel_axis: int = 3,
           filter_size: List[int] = None,
           stride: List[int] = None,
           padding: str = 'SAME',
           weight_norm: bool = True,
           initializer: Union[str, Callable[[Any], tf.Tensor]]
           = tf.initializers.glorot_normal,
           reuse: bool = False):
    """2D convolution for image

    Parameters
    ----------
    scope_name: str
        scope name
    x: tf.Tensor
        input tensor [B, H, W, C] or [B, C, H, W]
    n_out: int
        output channel size
    channel_axis: int
        channel's axis ex. if data_format==NHWC, 3
    filter_size: List[int]
        filter_size. default [3, 3]
    stride: List[int]
        stride. default [1, 1]
    padding: str
        padding type 'SAME' or 'VALID'
        see Tensorflow's official page
    weight_norm: bool
        apply weight normalization to weight tensor
    initializer: str or Callable
        initializer for weight tensor
    reuse: bool
        reuse variable

    Returns
    -------
    x: tf.Tensor
        convoluted x
        if you use default value and n_out == 64,
        output shape is [B, H, W, 64]

    Notes
    -----
    if x is tf.Tensor[B, H, W, C], channel_axis = 3,
    or x is tf.Tensor[B, C, H, W], channel_axis = 1
    """
    if stride is None:
        stride = [1, 1]
    if filter_size is None:
        filter_size = [3, 3]
    with tf.variable_scope('conv2d_' + scope_name, reuse=reuse):
        n_in = int(x.get_shape()[channel_axis])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, n_out]
        w = tf.get_variable('W',
                            filter_shape,
                            tf.float32,
                            initializer=initializer)
    with tf.name_scope('conv2d'):
        if weight_norm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if channel_axis == 3:
            x = tf.nn.conv2d(x, w, stride_shape, padding, data_format="NHWC")
        elif channel_axis == 1:
            x = tf.nn.conv2d(x, w, stride_shape, padding, data_format="NCHW")
        else:
            raise NotImplementedError("data format must be NHWC or NCHW")
        return x
