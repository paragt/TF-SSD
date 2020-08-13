import os
import importlib
import math
import numpy as np

import tensorflow as tf

#from source.network.detection 
import ssd_common

CLASS_WEIGHTS = 1.0
BBOXES_WEIGHTS = 1.0

'''
# Priorboxes
ANCHORS_STRIDE = [8, 16, 32, 64, 100, 300]
ANCHORS_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# control the size of the default square priorboxes
# REF: https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L164
MIN_SIZE_RATIO = 15 # 0.15 instead of 0.2
MAX_SIZE_RATIO = 90 # 0.9 as in SSD paper
INPUT_DIM = 300

ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,
                                                  ANCHORS_ASPECT_RATIOS,
                                                  MIN_SIZE_RATIO,
                                                  MAX_SIZE_RATIO,
                                                  INPUT_DIM)

'''


def encode_gt(inputs, batch_size):
  image_id, image, labels, boxes, scale, translation, file_name = inputs
  gt_labels, gt_bboxes, gt_masks = ssd_common.encode_gt(labels, boxes, ANCHORS_MAP, batch_size)
  return gt_labels, gt_bboxes, gt_masks


def ssd_feature(outputs, data_format):
    outputs_conv6_2 = ssd_common.ssd_block_bn(outputs, "conv6", data_format, [1, 2], [1, 3], [256, 512], ["SAME", "SAME"], activation='relu6')
    outputs_conv7_2 = ssd_common.ssd_block_bn(outputs_conv6_2, "conv7", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "SAME"], activation='relu6')
    outputs_conv8_2 = ssd_common.ssd_block_bn(outputs_conv7_2, "conv8", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "VALID"], activation='relu6')
    return outputs_conv6_2, outputs_conv7_2, outputs_conv8_2

    #outputs_conv9_2 = ssd_common.ssd_block(outputs_conv8_2, "conv9", data_format, [1, 2], [1, 3], [128, 256], ["SAME", "VALID"])

def net(image, #inputs,
        num_classes, NUM_ANCHORS,
        is_training,
        feature_net,
        feature_net_path='',
        data_format="channels_last"):

  #image_id, image, labels, boxes, scale, translation, file_name = inputs
  
  feature_net = getattr(
    importlib.import_module(feature_net),
    "net")

  feature_net_path = os.path.join(feature_net_path)

  print('format: ',data_format, ' is_train= ', is_training )
  _, endpts = feature_net(image, data_format=data_format, is_train=is_training, depth_mult=1.4) #1.0, 1.4


  with tf.variable_scope(name_or_scope='SSD',
                         values=[endpts],
                         reuse=tf.AUTO_REUSE):

    mb_feat8 = endpts['layer_8/expansion_output']
    mb_feat15 = endpts['layer_15/expansion_output']
    mb_feat19 = endpts['layer_19']

    # # Add shared features
    outputs_conv6_2, outputs_conv7_2, outputs_conv8_2 = ssd_feature(mb_feat19, data_format)

    classes = []
    bboxes = []
    feature_layers = (mb_feat8,   mb_feat15,    mb_feat19,  outputs_conv6_2,  outputs_conv7_2, outputs_conv8_2)
    name_layers =   ("MB/conv8e", "MB/conv15e", "MB/conv19", "SSD/conv6_2",   "SSD/conv7_2",   "SSD/conv8_2")

    for name, feat, num in zip(name_layers, feature_layers, NUM_ANCHORS):      
      classes.append(ssd_common.class_graph_fn(feat, num_classes, num, name))
      bboxes.append(ssd_common.bbox_graph_fn(feat, num, name))

    classes = tf.concat(classes, axis=1)
    bboxes = tf.concat(bboxes, axis=1)

    return classes, bboxes


def loss(gt, outputs):
  return ssd_common.loss(gt, outputs, CLASS_WEIGHTS, BBOXES_WEIGHTS)


def detect(feat_classes, feat_bboxes,ANCHORS_MAP, batch_size, num_classes, confidence_threshold):
  score_classes = tf.nn.softmax(feat_classes)

  feat_bboxes = ssd_common.decode_bboxes_batch(feat_bboxes, ANCHORS_MAP, batch_size)

  detection_topk_scores, detection_topk_labels, detection_topk_bboxes, detection_topk_anchors = ssd_common.detect_batch(
    score_classes, feat_bboxes, ANCHORS_MAP, batch_size, num_classes, confidence_threshold)

  return detection_topk_scores, detection_topk_labels, detection_topk_bboxes,detection_topk_anchors
