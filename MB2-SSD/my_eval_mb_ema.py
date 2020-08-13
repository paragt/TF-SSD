import os,sys
import argparse
import pdb
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import time

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow.python.tools import inspect_checkpoint as chkp

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from my_config_mb import *
import ssd_augmenter_mb
import ssd300_mb, ssd_common
import preprocess
import dataset_feeder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/tf_objdet/trn_tf_script/data_COCO/')
parser.add_argument('--dataset_meta', default='val2014')
parser.add_argument('--mode', default='eval')#'train')
parser.add_argument('--minival_ids_file', default=None)#'train')
parser.add_argument('--resolution', default=300, type=int)
parser.add_argument('--resize_method', default='BILINEAR')
parser.add_argument('--smallest_ratio', default="4,10")
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
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


def ema_to_weights(ema):
    return tf.group(*(tf.assign(var, ema.average(var).read_value())
                     for var in tf.trainable_variables()))

def main(args):

    #pdb.set_trace()
    mscoco = dataset_feeder.MscocoFeeder(args, args.dataset_meta[0])
    nsamples = len(mscoco.samples)
    nclasses = len(mscoco.cat_names)+ 1 #categories + background
    batch_size_per_gpu = args.batch_size
    batch_size = args.num_gpus*batch_size_per_gpu
    pretrained_model_path = args.pretrained_model_path 
    smallest_ratio = [int(x) for x in args.smallest_ratio.split(',')]
    tfobj = preprocess.TFdata(args, nsamples, ssd_augmenter_mb.augment)

    if args.minival_ids_file is not None:
        minival_ids = np.loadtxt(args.minival_ids_file, dtype=int).tolist()

    ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,ANCHORS_ASPECT_RATIOS,MIN_SIZE_RATIO, MAX_SIZE_RATIO, INPUT_DIM, smallest_ratio)
    #get_sample = mscoco.get_samples_fn()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        dataset = tfobj.create_tfdataset([mscoco], batch_size, args.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image_id, images, classes, boxes, scale, translation, filename = iterator.get_next()

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        split_images=tf.split(images, args.num_gpus)
        split_classes = tf.split(classes, args.num_gpus)
        split_boxes = tf.split(boxes, args.num_gpus)

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

        ema_op = ema.apply([v for v  in tf.trainable_variables()])

        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        #variables_to_restore = [v for v  in tf.trainable_variables()]
        #other_variables = [v for v in tf.global_variables() if (v not in variables_to_restore and 'batch_normalization' in v.name and 'moving' in v.name)]
        #variables_to_restore.extend(other_variables)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_model_path,  variables_to_restore)


    #pdb.set_trace()
    with tf.Session(graph=graph) as sess:

        #chkp.print_tensors_in_checkpoint_file(pretrained_model_path, tensor_name='',all_tensors=False, all_tensor_names=True)

        sess.run(global_init)
        var_load_fn(sess)
 
        sess.run(ema_to_weights(ema))
        
        '''
        variables_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, pretrained_model_path)
        '''
        start_time = time.time()

        #classes , boxes, split_classes, split_boxes = sess.run([classes,boxes,split_classes,split_boxes])
        nbatches = int(nsamples/batch_size)
        
        all_detections = []
        all_image_ids = []
        for bb in range(nbatches):
            #if bb>0 and (bb%100==0): print('batch = ',bb) 
            image_ids,imgnames,scores,classes,boxes = sess.run([image_id, filename,topk_scores, topk_labels, topk_bboxes]) 
            
            num_images = len(imgnames)
            for i in range(num_images):
                file_name = imgnames[i][0].decode('ascii')
                if (args.minival_ids_file is not None) and (image_ids[i][0] not in minival_ids):
                    continue
  
                # print(file_name)
                num_detections = len(classes[i])

                input_image = np.array(Image.open(file_name))
                h, w = input_image.shape[:2]

                # COCO evaluation is based on per detection
                for d in range(num_detections):
                    box = boxes[i][d]
                    box = box * [float(w), float(h), float(w), float(h)]
                    box[0] = np.clip(box[0], 0, w)
                    box[1] = np.clip(box[1], 0, h)
                    box[2] = np.clip(box[2], 0, w)
                    box[3] = np.clip(box[3], 0, h)
                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]
                    result = {
                        "image_id": image_ids[i][0],
                        "category_id": mscoco.class_id_to_category_id[classes[i][d]],
                        "bbox": box.tolist(),
                        "score": scores[i][d]
                    }
                    all_detections.append(result)

                all_image_ids.append(image_ids[i][0])


        elapsed_time = time.time() - start_time
        print('elapsed time = '+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


        #pdb.set_trace()
        #fid=open('test_detections.json','wt')
        #json.dump(all_detections, fid)
        #fid.close()            
        print('batch = ',bb)
        print('Finished prediction ... ')
        if len(all_detections) > 0:
            annotation_file = os.path.join(args.dataset_dir, "annotations","instances_" + args.dataset_meta[0] + ".json") #"instances_" + DATASET_META + ".json")
            coco = COCO(annotation_file)

            coco_results = coco.loadRes(all_detections)

            cocoEval = COCOeval(coco, coco_results, "bbox")
            cocoEval.params.imgIds = all_image_ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()



if __name__=='__main__':
    args = parser.parse_args()
    args.dataset_meta=args.dataset_meta.split(',')
    main(args)
