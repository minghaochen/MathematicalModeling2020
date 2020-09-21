# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:59:53 2020

@author: mhchen
"""

def guess_label(self, y, classifier, T, **kwargs):
    del kwargs
    logits_y = [classifier(yi, training=True) for yi in y]
    logits_y = tf.concat(logits_y, 0)
    # Compute predicted probability distribution py.
    p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
    p_model_y = tf.reduce_mean(p_model_y, axis=0)
    # Compute the target distribution.
    p_target = tf.pow(p_model_y, 1. / T)
    p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
    return EasyDict(p_target=p_target, p_model=p_model_y)

def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', **kwargs):
    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
    l_in = tf.placeholder(tf.int32, [None], 'labels')
    wd *= lr
    w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
    augment = MixMode(mixmode)
    classifier = functools.partial(self.classifier, **kwargs)

    y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
    guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
    ly = tf.stop_gradient(guess.p_target)
    lx = tf.one_hot(l_in, self.nclass)
    xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
    x, y = xy[0], xy[1:]
    labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
    del xy, labels_xy

    batches = layers.interleave([x] + y, batch)
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits = [classifier(batches[0], training=True)]
    post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
    for batchi in batches[1:]:
        logits.append(classifier(batchi, training=True))
    logits = layers.interleave(logits, batch)
    logits_x = logits[0]
    logits_y = tf.concat(logits[1:], 0)

    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
    loss_l2u = tf.reduce_mean(loss_l2u)
    tf.summary.scalar('losses/xe', loss_xe)
    tf.summary.scalar('losses/l2u', loss_l2u)

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.append(ema_op)
    post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

    train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
        train_op = tf.group(*post_ops)

    # Tuning op: only retrain batch norm.
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    classifier(batches[0], training=True)
    train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                          if v not in skip_ops])

    return EasyDict(
        x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
        classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
        classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))