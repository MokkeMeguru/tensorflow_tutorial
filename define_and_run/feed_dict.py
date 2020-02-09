import argparse

import tensorflow as tf


class TensorflowHelloFeedDict():
    """
    attributes:
    - name: tf.Tensor
    developer's name (placeholder)
    """

    def __init__(self):
        self.name = tf.placeholder(tf.string)
        self.greeting = tf.print(tf.strings.format(
            'Hello, Tensorflow! --{}', self.name))

    def __call__(self, sess: tf.Session, name: str = 'Jack'):
        sess.run(self.greeting, feed_dict={self.name: name})


def parse():
    """Parse Args
    note:
    in ipython, it don't use argparse
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--name', help='your name', default='Jack')
    if hasattr(__builtins__, '__IPYTHON__'):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse()
    print('[Info] initialize')
    tf_hello = TensorflowHelloFeedDict()
    print('[Info] Call')
    with tf.Session() as sess:
        tf_hello(sess, args.name)


if __name__ == '__main__':
    main()
