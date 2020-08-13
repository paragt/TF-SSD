from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib import layers
#from tensorflow.contrib.framework.python.ops import arg_scope
#from tensorflow.contrib.layers.python.layers import layers as layers_lib
#from tensorflow.contrib.layers.python.layers import regularizers
#from tensorflow.contrib.layers.python.layers import utils
#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import init_ops
#from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops import variable_scope
#import tensorflow.contrib.slim as slim



def conv_block(infeat, ntimes, nfeat, prefix):
    ret = infeat 
    for ii in range(ntimes):
        vname = prefix+'/'+prefix+'_'+str(ii+1)
        ret = tf.layers.conv2d(ret, nfeat, [3, 3], name=vname, padding='SAME', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))

    return ret

def vgg_16(inputs,  scope='vgg_16'):

  endpoints={}
  #with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
  with tf.variable_scope(name_or_scope=scope, values=[inputs], reuse=tf.AUTO_REUSE):
      #net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = conv_block(inputs, 2, 64, 'conv1')
      net = tf.layers.max_pooling2d(net, pool_size = [2, 2], strides=[2, 2], padding = 'SAME', name='pool1')
      #net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = conv_block(net, 2, 128, 'conv2')
      net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding = 'SAME', name ='pool2')
      #net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = conv_block(net, 3, 256, 'conv3')
      net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding = 'SAME', name='pool3')
      #net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = conv_block(net, 3, 512, 'conv4')
      endpoints['conv4_3'] = net
      conv4_3out = net
      net = tf.layers.max_pooling2d(net, pool_size=[2, 2],strides=[2, 2], padding = 'SAME', name='pool4')
      #outputs_pool4 = net
      #net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = conv_block(net, 3, 512, 'conv5')

      endpoints['conv5_3'] = net
      # modified layers
      net = tf.layers.max_pooling2d(net,pool_size=[3, 3],strides=[1, 1],padding="SAME",name='pool5') 
      #outputs_pool5 = net
      endpoints['pool5'] = net

      # dilate 3,3 for pool5 w/ stride 2; dilate 6, for stride 1
      net = tf.layers.conv2d(net, 1024, [3, 3], strides=[1, 1], dilation_rate=[6,6],  padding='SAME', name = 'fc6',activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))
      endpoints['fc6'] = net

      net = tf.layers.conv2d(net, 1024, [1, 1], strides=[1, 1], padding='SAME', name = 'fc7',activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))
      endpoints['fc7'] = net
      fc7out = net

      return [conv4_3out, fc7out]
      #return [endpoints['conv4_3'], endpoints['fc7']]


def net(inputs,  data_format='channels_last',  VGG_PARAMS_FILE= None, is_train=False):

    if data_format != "channels_last":
        print('only works for channels last now')
        return None
    outputs = vgg_16(inputs)
    return outputs

