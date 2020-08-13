import os,sys
import argparse
import pdb
import tensorflow as tf
import numpy as np
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')


from my_config import *
import ssd_augmenter_vgg
import ssd300, ssd_common
import dataset_feeder
import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/tf_objdet/trn_tf_script/data_COCO/')
parser.add_argument('--dataset_meta', default='val2014')
parser.add_argument('--mode', default='eval')#'train')
parser.add_argument('--resolution', default=300, type=int)
parser.add_argument('--resize_method', default='BILINEAR')
parser.add_argument('--smallest_ratio', default="4,10")
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--pretrained_model_path', default='', type=str)
parser.add_argument('--confidence_threshold', default=0.3, type=float)
'''
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--epoch_size', default=0, type=int)
parser.add_argument('--cost_balance', default=10., type=float)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float) # google  uses 4e-5, was using 5e-4 before
'''

def create_tfdataset(feeder, batch_size, num_epochs):

    dataset = tf.data.Dataset.from_generator(generator=lambda: feeder.get_samples_fn(), output_types=(tf.int64, tf.string, tf.int64, tf.float32))

    #dataset = dataset.shuffle(feeder.get_num_samples())

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(lambda image_id, file_name, classes, boxes: feeder.parse_fn(image_id, file_name, classes, boxes), num_parallel_calls=12)

    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None, None, 3], [None], [None, 4], [None], [None], [None])) # id, image, classes, boxes, scale, translation

    dataset = dataset.prefetch(2)
    return dataset

def main(args):

    #pdb.set_trace()
    mscoco = dataset_feeder.MscocoFeeder(args,args.dataset_meta[0])
    nsamples = len(mscoco.samples)
    nclasses = len(mscoco.cat_names)+ 1 #categories + background
    batch_size = args.num_gpus*args.batch_size
    pretrained_model_path = args.pretrained_model_path 
    tfobj = preprocess.TFdata(args, nsamples, ssd_augmenter_vgg.augment)

    smallest_ratio = [int(x) for x in args.smallest_ratio.split(',')]
    ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,ANCHORS_ASPECT_RATIOS,MIN_SIZE_RATIO, MAX_SIZE_RATIO, INPUT_DIM,smallest_ratio)
    #get_sample = mscoco.get_samples_fn()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        dataset = tfobj.create_tfdataset([mscoco], batch_size, args.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image_id, images, classes, boxes, scale, translation, filename = iterator.get_next()

        split_images=tf.split(images, args.num_gpus)
        split_classes = tf.split(classes, args.num_gpus)
        split_boxes = tf.split(boxes, args.num_gpus)

        pred_classes_array= []
        pred_boxes_array= []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.name_scope('tower_%d' % (i)), tf.device('/gpu:%d' % i):
                    pred_classes, pred_boxes  = ssd300.net(split_images[i], nclasses, NUM_ANCHORS, False, 'my_vgg3')
                    pred_classes_array.append(pred_classes)
                    pred_boxes_array.append(pred_boxes)
                    tf.get_variable_scope().reuse_variables()


        pred_classes_array = tf.concat(values=pred_classes_array,axis=0)
        pred_boxes_array = tf.concat(values=pred_boxes_array,axis=0)

        topk_scores, topk_labels, topk_bboxes, _ = ssd300.detect(pred_classes_array, pred_boxes_array, ANCHORS_MAP, batch_size, nclasses, args.confidence_threshold)

        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        #variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['resnet_model/dense','global_step'])
        variables_to_restore = [v for v  in tf.trainable_variables()]
        #other_variables = [v for v in tf.global_variables() if (v not in variables_to_restore and 'batch_normalization' in v.name and 'moving' in v.name)]
        #variables_to_restore.extend(other_variables)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_model_path,  variables_to_restore)


    #pdb.set_trace()
    with tf.Session(graph=graph) as sess:
        print('Start counting time ..')
        sess.run(global_init)
        var_load_fn(sess)
        #classes , boxes, split_classes, split_boxes = sess.run([classes,boxes,split_classes,split_boxes])
        nbatches = int(nsamples/batch_size)
        all_det = {}
        for bb in range(nbatches):
            #print('batch = ',bb) 
            imgname,scores,classes,boxes = sess.run([filename,topk_scores, topk_labels, topk_bboxes]) 
            for ii,iname in enumerate(imgname):
                iname = iname[0].decode('ascii')
                cat_id = [mscoco.class_id_to_category_id[v] for v in classes[ii].tolist()]
                cat_name = [mscoco.catid_to_name[v] for v in cat_id]
                all_det[iname] = {'class_name':cat_name, 'score':scores[ii].tolist(), 'box':boxes[ii].tolist()}

        output_name = 'detections_'+args.dataset_meta[0]+'.json'
        with open(output_name,'wt') as fid:
            json.dump(all_det, fid)
if __name__=='__main__':
    args = parser.parse_args()
    args.dataset_meta=args.dataset_meta.split(',')
    main(args)
