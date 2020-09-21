# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:02:22 2020

@author: mhchen
"""

def model(self, lr, wd, ema, **kwargs):
    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
    l_in = tf.placeholder(tf.int32, [None], 'labels')
    wd *= lr
    classifier = functools.partial(self.classifier, **kwargs)

    def get_logits(x):
        logits = classifier(x, training=True)
        return logits

    x, labels_x = self.augment(x_in, tf.one_hot(l_in, self.nclass), **kwargs)
    logits_x = get_logits(x)
    post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    y, labels_y = self.augment(y_in, tf.nn.softmax(get_logits(y_in)), **kwargs)
    labels_y = tf.stop_gradient(labels_y)
    logits_y = get_logits(y)

    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_xeu = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_y, logits=logits_y)
    loss_xeu = tf.reduce_mean(loss_xeu)
    tf.summary.scalar('losses/xe', loss_xe)
    tf.summary.scalar('losses/xeu', loss_xeu)

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.append(ema_op)
    post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

    train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + loss_xeu, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
        train_op = tf.group(*post_ops)

    # Tuning op: only retrain batch norm.
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    classifier(x_in, training=True)
    train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                          if v not in skip_ops])

    return EasyDict(
        x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
        classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
        classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))
