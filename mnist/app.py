from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import model
import params
import params_utils


class args:
    task = 'test'
    prefix = 'tmp'


hps = params_utils.parse_params(params.mnist_classification_args)

# open interactive session
sess = tf.InteractiveSession()

# inputs setting
with tf.variable_scope('input'):
    if hps.data_params.is_flattened:
        shape = [None, np.prod(hps.data_params.image_size)]
    else:
        shape = [None] + list(hps.data_params.image_size)
    x = tf.placeholder(tf.float32, shape)
    y = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, shape=None)

# format image
if hps.data_params.is_flattened:
    if len(hps.data_params.image_size) == 3:
        image_shape = [-1] + list(hps.data_params.image_size)
    elif len(hps.data_params.image_size) == 2:
        image_shape = [-1] + list(hps.data_params.image_size) + [1]
    else:
        raise NotImplementedError('image shape should be NHW or NHWC')
_x = tf.reshape(x, image_shape)


# input -> model -> output
y_hat = model.model_fn(_x, hps, is_training)

saver = tf.train.Saver()
path_prefix = Path(args.prefix)
save_path = hps.paths.model_path / path_prefix
print(save_path)
ckpt = tf.train.get_checkpoint_state(save_path)

if ckpt:
    print('restore variable')
    last_model = ckpt.model_checkpoint_path
    saver.restore(sess, last_model)
else:
    raise Exception('not found')


app = Flask(__name__)


@app.route('/api/mnist', method=['GET'])
def mnist_classification():
    img = (np.array(request.json, dtype=uint8) / 255.0 - 0.5)
    predict = sess.run(y_hat, feed_dict={x: img})
    predict_label = tf.argmax(predict)
    return jsonify(result=[predict_label])


if __name__ == '__main__':
    app.run()
    sess.close()
