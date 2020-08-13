
import tensorflow as tf
from mobilenet import mobilenet_v2


def net(inputs,  data_format='channels_last', depth_mult=1.0,  VGG_PARAMS_FILE= None, is_train=False):

    if data_format != "channels_last":
        print('only works for channels last now')
        return None

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_train)):
        logits, endpoints = mobilenet_v2.mobilenet(inputs, base_only=True, reuse=tf.AUTO_REUSE, final_endpoint="layer_19", depth_multiplier=depth_mult)

    l15e = endpoints['layer_15/expansion_output']
    l19 = endpoints['layer_19']

    return [l15e, l19], endpoints

