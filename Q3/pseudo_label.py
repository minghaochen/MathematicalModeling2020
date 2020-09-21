# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:01:37 2020

@author: mhchen
"""

def model(self, lr, wd, ema, warmup_pos, consistency_weight, threshold, **kwargs):
    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
    l_in = tf.placeholder(tf.int32, [None], 'labels')
    l = tf.one_hot(l_in, self.nclass)
    wd *= lr
    warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

    classifier = functools.partial(self.classifier, **kwargs)
    logits_x = classifier(x_in, training=True)
    post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
    logits_y = classifier(y_in, training=True)
    # Get the pseudo-label loss
    loss_pl = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(logits_y, axis=-1), logits=logits_y
    )
    # Masks denoting which data points have high-confidence predictions
    greater_than_thresh = tf.reduce_any(
        tf.greater(tf.nn.softmax(logits_y), threshold),
        axis=-1,
        keepdims=True,
    )
    greater_than_thresh = tf.cast(greater_than_thresh, loss_pl.dtype)
    # Only enforce the loss when the model is confident
    loss_pl *= greater_than_thresh
    # Note that we also average over examples without confident outputs;
    # this is consistent with the realistic evaluation codebase
    loss_pl = tf.reduce_mean(loss_pl)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('losses/xe', loss)
    tf.summary.scalar('losses/pl', loss_pl)

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.append(ema_op)
    post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

    train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_pl * warmup * consistency_weight,
                                                   colocate_gradients_with_ops=True)
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
