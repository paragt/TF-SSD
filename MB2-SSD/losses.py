import tensorflow as tf
from my_config_mb import *
import ssd_common

def compute_loss(pred_classes, pred_boxes, split_classes, split_boxes, split_mask, error_normalize_factor=1.0, huber_loss_delta=1.0, weight_decay=1e-4):

    gt_mask = tf.reshape(split_mask, [-1])
    logits_classes = tf.reshape(pred_classes, [-1, tf.shape(pred_classes)[2]])
    gt_classes = tf.reshape(split_classes, [-1, 1])
    logits_bboxes = tf.reshape(pred_boxes, [-1, 4])
    gt_bboxes = tf.reshape(split_boxes, [-1, 4])
    num_cls = tf.shape(logits_classes)[-1]

    fg_index, bg_index = ssd_common.hard_negative_mining(logits_classes, gt_mask)

    fg_labels0 = tf.gather(gt_classes, fg_index)
    bg_labels0 = tf.gather(gt_classes, bg_index)

    fg_logits0 = tf.gather(logits_classes, fg_index)
    bg_logits0 = tf.gather(logits_classes, bg_index)


    fg_logits = tf.reshape(fg_logits0,[-1, num_cls])
    fg_labels = tf.reshape(fg_labels0,[-1])
    bg_logits = tf.reshape(bg_logits0,[-1, num_cls])
    bg_labels = tf.reshape(bg_labels0,[-1])

    nmatches = tf.cast(tf.shape(fg_logits)[0],tf.float32)

    fg_cls_loss = tf.cond(tf.less(nmatches,1), lambda: 0.0, lambda: tf.losses.sparse_softmax_cross_entropy(labels=fg_labels, logits=fg_logits,reduction=tf.losses.Reduction.SUM))
    bg_cls_loss = tf.cond(tf.less(nmatches,1), lambda: 0.0, lambda:tf.losses.sparse_softmax_cross_entropy(labels=bg_labels, logits=bg_logits,reduction=tf.losses.Reduction.SUM))

    pred_bboxes_select = tf.gather(logits_bboxes, fg_index)
    gt_bboxes_select = tf.gather(gt_bboxes, fg_index)

    pred_bboxes_select = tf.reshape(pred_bboxes_select,[-1,4])
    gt_bboxes_select = tf.reshape(gt_bboxes_select,[-1,4])

    bboxes_loss = tf.cond(tf.less(nmatches,1), lambda: 0.0, lambda:tf.losses.huber_loss(gt_bboxes_select, pred_bboxes_select, delta=huber_loss_delta))
    nnegs = tf.cast(tf.shape(bg_logits)[0],tf.float32)

    class_loss = tf.cond(tf.less(nmatches,1), lambda: 0.0, lambda : (fg_cls_loss + bg_cls_loss)/(error_normalize_factor*nmatches))

    l2_var_list = [v for v in tf.trainable_variables() if not any(x in v.name for x in SKIP_L2_LOSS_VARS)]

    loss_l2 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in l2_var_list])

    loss_ = tf.cond(tf.less(nmatches,1), lambda: 0.0, lambda: class_loss + bboxes_loss + loss_l2)

    return loss_, class_loss, bboxes_loss, loss_l2, nmatches, nnegs
