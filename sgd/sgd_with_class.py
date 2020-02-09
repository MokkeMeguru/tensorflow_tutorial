import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


class SimpleModel():
    """
    attributes:
    - scope: str
    variable's scope
    """

    def build(self):
        scope = self.scope + '/weights'
        with tf.variable_scope(scope):
            self.a = tf.get_variable(
                name=scope + '_a', shape=[], dtype=tf.float32,
                initializer=tf.initializers.constant(
                    value=0.0, dtype=tf.float32),
                trainable=True)
            self.b = tf.get_variable(
                name=scope + '_b', shape=[], dtype=tf.float32,
                initializer=tf.initializers.constant(
                    value=0.0, dtype=tf.float32),
                trainable=True)

    def __init__(self, scope: str = 'simple_model'):
        self.scope = scope
        self.build()

    def __call__(self, x):
        with tf.name_scope(self.scope + '/formula'):
            y = self.a * x + self.b
        return y


class SGD_Task():
    """
    attributes:
    - scope: str
    variable's scope
    """

    def build(self):
        with tf.variable_scope(self.scope + '/inputs'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[])
            self.y = tf.placeholder(dtype=tf.float32, shape=[])

    def __init__(self, scope: str = 'sgd', path: Path = './tmp/sgd_class'):
        self.path = path
        self.lr = 1e-3
        self.scope = scope
        self.model = SimpleModel()
        self.build()
        with tf.name_scope(self.scope + '/operations'):
            self.init = tf.global_variables_initializer()
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.lr)
            self.loss = tf.math.squared_difference(self.y, self.model(self.x))
            self.train_step = self.optimizer.minimize(self.loss)
            self.logs = tf.print(tf.strings.format(
                'a - {} / b - {} / loss - {}',
                [self.model.a, self.model.b, self.loss]))

    def init_summary(self, sess: tf.Session):
        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path, sess.graph)

    def teacher(self, x):
        return 5.0 * x + 8.0

    def train(self):
        with tf.Session() as sess:
            sess.run(self.init)
            self.init_summary(sess)
            for step in range(1000):
                x = np.random.uniform()
                y = self.teacher(x)
                feed_dict = {
                    self.x: x,
                    self.y: y}
                if step % 100 == 0:
                    summary, _, _ = sess.run(
                        [self.summary, self.train_step, self.logs], feed_dict)
                else:
                    summary, _ = sess.run(
                        [self.summary, self.train_step], feed_dict)
                self.writer.add_summary(summary, step)

def parse():
    """Parse Args
    note:
    in ipython, it don't use argparse
    """
    return None

def main():
    args = parse()
    sgd = SGD_Task()
    sgd.train()


if __name__ == '__main__':
    main()
