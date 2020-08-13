import os,sys,glob
import argparse
import pdb
import tensorflow as tf
import numpy as np
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from my_config_mb import *
import ssd_augmenter_mb
import ssd300_mb, ssd_common
import dataset_feeder
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/tf_objdet/trn_tf_script/data_COCO/')
parser.add_argument('--dataset_meta', default='val2014')
parser.add_argument('--exclude_img_list', default='none')
parser.add_argument('--category_map_file', default='none')
parser.add_argument('--mode', default='eval')#'train')
parser.add_argument('--resolution', default=300, type=int)
parser.add_argument('--resize_method', default='BILINEAR')
parser.add_argument('--smallest_ratio', default="4,10")
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--pretrained_model_path', default='', type=str)
parser.add_argument('--confidence_threshold', default=0.01, type=float)
parser.add_argument('--output', type=str)

'''
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--epoch_size', default=0, type=int)
parser.add_argument('--cost_balance', default=10., type=float)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float) # google  uses 4e-5, was using 5e-4 before
'''

def main(args):

    pdb.set_trace()
    mscoco = dataset_feeder.MscocoFeeder(args,args.dataset_meta[0],category_map_file=args.category_map_file)
    nsamples = len(mscoco.samples)
    nclasses = len(mscoco.cat_names)+ 1 #categories + background
    batch_size = args.num_gpus*args.batch_size
    pretrained_model_path = args.pretrained_model_path 

    tfobj = preprocess.TFdata(args, nsamples, ssd_augmenter_mb.preprocess_for_apply)

    smallest_ratio = [int(x) for x in args.smallest_ratio.split(',')]
    ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,ANCHORS_ASPECT_RATIOS,MIN_SIZE_RATIO, MAX_SIZE_RATIO, INPUT_DIM,smallest_ratio)
    #get_sample = mscoco.get_samples_fn()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        input_names = tf.placeholder(dtype=tf.string, name='input_names')
        #images = tfobj.parse_fn(0, input_names, [0], tf.zeros([1,4]))

        preprocess_fn  = lambda x: tfobj.parse_fn_apply(x)
        images = tf.map_fn(preprocess_fn, input_names, dtype=tf.float32)
        
        
        split_images=tf.split(images, args.num_gpus)
        
        pred_classes_array= []
        pred_boxes_array= []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.name_scope('tower_%d' % (i)), tf.device('/gpu:%d' % i):
                    pred_classes, pred_boxes  = ssd300_mb.net(split_images[i], nclasses, NUM_ANCHORS, False, 'my_mb')
                    pred_classes_array.append(pred_classes)
                    pred_boxes_array.append(pred_boxes)
                    tf.get_variable_scope().reuse_variables()


        pred_classes_array = tf.concat(values=pred_classes_array,axis=0)
        pred_boxes_array = tf.concat(values=pred_boxes_array,axis=0)

        topk_scores, topk_labels, topk_bboxes, _ = ssd300_mb.detect(pred_classes_array, pred_boxes_array, ANCHORS_MAP, batch_size, nclasses, args.confidence_threshold)

        

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        #variables_to_restore = [v for v  in tf.trainable_variables()]
        #other_variables = [v for v in tf.global_variables() if (v not in variables_to_restore and 'batch_normalization' in v.name and 'moving' in v.name)]
        #variables_to_restore.extend(other_variables)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_model_path,  variables_to_restore)
         
        global_init=tf.global_variables_initializer() # initializes all variables in the graph
     
    pdb.set_trace()
    inputfolder = '/home/testing_framework/check_testing_website/motion_1frame_max'
    subfolders = os.listdir(inputfolder)
    all_det = []
    with tf.Session(graph=graph) as sess:
        print('Start counting time ..')
        sess.run(global_init)
        var_load_fn(sess)
        for subf in subfolders:
            img_filenames = glob.glob(os.path.join(inputfolder,subf,'*.jpg'))
            for input_name in img_filenames:
                scores,classes,boxes = sess.run([topk_scores, topk_labels, topk_bboxes], feed_dict={input_names:[input_name]})
                dets = [] 
                for ii in range(len(scores[0])):
                    if scores[0][ii] < args.confidence_threshold:
                        continue
                    det1 = {}
                    cat_id = mscoco.class_id_to_category_id[classes[0][ii]]
                    cat_name = mscoco.catid_to_name[cat_id]
                    det1['score'] = float(scores[0][ii])
                    detected_box = np.clip(boxes[0][ii],0,1)
                    det1['box'] = detected_box.tolist()
                    det1['name'] = cat_name
                    if cat_name=='cat' or cat_name=='dog':
                        det1['name']='pet'
                    dets.append(det1)
          
                all_det.append({'Id':subf, 'Image':os.path.basename(input_name), 'Detections':dets})
                print(dets)

        pdb.set_trace()
        output_name = args.output
        with open(output_name,'wt') as fid:
            json.dump(all_det, fid)
if __name__=='__main__':
    args = parser.parse_args()
    args.dataset_meta=args.dataset_meta.split(',')
    main(args)
