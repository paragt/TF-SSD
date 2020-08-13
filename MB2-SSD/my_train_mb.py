import os,sys
import argparse
import pdb
import tensorflow as tf
import numpy as np
import json
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from my_config_mb import *
import dataset_feeder
import ssd_augmenter_mb
import ssd300_mb, ssd_common
from draw_detections import mark_detections
from multigpu import average_gradients
import losses
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/tf_objdet/trn_tf_script/data_COCO/')
parser.add_argument('--dataset_meta', default='train2014')
parser.add_argument('--exclude_img_list', default='none')
parser.add_argument('--category_map_file', default='none')
parser.add_argument('--mode', default='train')#'train')
parser.add_argument('--resize_method', default='BILINEAR', choices=['BILINEAR', 'ASPECT', 'PAD', 'NONE'])
parser.add_argument('--lr_steps',    default=     "1000, 2000,  70000 , 140000 ,  210000, 280000, 350000")
parser.add_argument('--lr_values', default="0.00001, 0.0001, 0.009,0.003,  0.0009,  0.0003, 0.00009,  0.00003")
parser.add_argument('--smallest_ratio', default="4,10") # 7,15
parser.add_argument('--resolution', default=300, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--num_gpus', default=2, type=int)
parser.add_argument('--pretrained_feature_path', default='', type=str)
parser.add_argument('--confidence_threshold', default=0.3, type=float)
parser.add_argument('--error_normalize_factor', default=2.0, type=float)
parser.add_argument('--learning_rate', default=0.005, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float) 
parser.add_argument('--ema_decay', default=0.9999, type=float) 
parser.add_argument('--huber_loss_delta', default=1.0, type=float) 
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--use_bckgnd', default='False', choices=['True', 'False'])
'''
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--epoch_size', default=0, type=int)
parser.add_argument('--cost_balance', default=10., type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
'''


def ema_to_weights(ema):
    return tf.group(*(tf.assign(var, ema.average(var).read_value())
                     for var in tf.trainable_variables()))

def get_lr_optimizer(step, boundaries, values):

    learning_rate_piecewise = tf.train.piecewise_constant(step, boundaries, values)

    #decay_step_exp = 15000
    #learning_rate_exp = tf.train.exponential_decay(init_lr, step, decay_step_exp, 0.8, staircase=True)

    learning_rate = learning_rate_piecewise
    tf.summary.scalar('lr',learning_rate)

    #tf.summary.scalar('epsilon',epsilon)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=False)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,epsilon=1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)
   
    return learning_rate, optimizer     


def get_var_load_fn(last_model, pretrained_feature_path):

    if last_model is not None:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[])
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(last_model,  variables_to_restore)
    elif pretrained_feature_path is not None:
        variables_to_restore = [v for v  in tf.trainable_variables() if 'MobilenetV2' in v.name]
        other_variables = [v for v in tf.global_variables() if (v not in variables_to_restore and 'MobilenetV2' in v.name and 'BatchNorm' in v.name and 'moving' in v.name)]
        variables_to_restore += other_variables
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_feature_path,  variables_to_restore)

    return var_load_fn



def main(args):


    lr_boundaries = [int(x) for x in args.lr_steps.split(',')]
    lr_values = [float(x) for x in args.lr_values.split(',')]


    smallest_ratio = [int(x) for x in args.smallest_ratio.split(',')]
    ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,ANCHORS_ASPECT_RATIOS,MIN_SIZE_RATIO, MAX_SIZE_RATIO, INPUT_DIM, smallest_ratio)

    dtst_list=[]
    for ii,dtst_name in enumerate(args.dataset_meta): 
        mscoco1 = dataset_feeder.MscocoFeeder(args, dtst_name, exclude_img_list_file=args.exclude_img_list[ii], category_map_file=args.category_map_file)
        dtst_list.append(mscoco1)

    nsamples = sum([v.num_samples for v in dtst_list])

    tfobj = preprocess.TFdata(args, nsamples, ssd_augmenter_mb.augment)

    #mscoco = dataset_feeder.MscocoFeeder(args,ssd_augmenter_vgg.augment)
    nclasses = len(dtst_list[0].cat_names)+ 1 #categories + background
    batch_size_per_gpu = args.batch_size
    batch_size_total = args.num_gpus*batch_size_per_gpu
    pretrained_feature_path = args.pretrained_feature_path 
    exm_per_epoch = nsamples
    modelDir = args.model_dir

    last_model, last_step = check_for_existing_model(modelDir)
    nu_epoch0 = exm_per_epoch*1.0/(batch_size_per_gpu*args.num_gpus)


    #pdb.set_trace()
    #get_sample = mscoco.get_samples_fn()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        step=tf.train.get_or_create_global_step()
        nu_epoch = tf.cast(nu_epoch0, tf.int32)
        init_lr = args.learning_rate

        boundaries = lr_boundaries
        values = lr_values
        learning_rate, optimizer = get_lr_optimizer(step, boundaries, values)        


        exp_mov_avg = tf.train.ExponentialMovingAverage(decay=args.ema_decay)

        dataset = tfobj.create_tfdataset(dtst_list, batch_size_total, args.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image_id, images, classes, boxes, scale, translation, filename = iterator.get_next()
        gt_classes, gt_boxes, gt_mask = ssd_common.encode_gt(classes,boxes, ANCHORS_MAP, batch_size_total )
        
        split_images=tf.split(images, args.num_gpus)
        split_classes = tf.split(gt_classes, args.num_gpus)
        split_boxes = tf.split(gt_boxes, args.num_gpus)
        split_mask = tf.split(gt_mask, args.num_gpus)

        pred_classes_array= []
        pred_boxes_array= []
        tower_grads = []
        class_loss_array = []
        box_loss_array = []
        l2_loss_array = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.name_scope('tower_%d' % (i)), tf.device('/gpu:%d' % i):
                    pred_classes, pred_boxes  = ssd300_mb.net(split_images[i], nclasses, NUM_ANCHORS, True, 'my_mb')

                    loss, class_loss, bboxes_loss, loss_l2, nmatches, nnegs = losses.compute_loss(pred_classes, pred_boxes, split_classes[i], split_boxes[i], split_mask[i], huber_loss_delta=args.huber_loss_delta, error_normalize_factor=args.error_normalize_factor, weight_decay=args.weight_decay)

                    loss = tf.identity( loss , "total_loss")

                    grads = optimizer.compute_gradients(loss) #, var_list=self.train_vars)
                    #loss = tf.identity(class_loss + loss_l2, "total_loss")
                    #grads = optimizer.compute_gradients(loss, var_list=[v for v in tf.trainable_variables() if 'bbox' not in v.name])

                    tower_grads.append(grads)

                    pred_classes_array.append(pred_classes)
                    pred_boxes_array.append(pred_boxes)
                    class_loss_array.append(class_loss)
                    box_loss_array.append(bboxes_loss)
                    l2_loss_array.append(loss_l2)
                    
                    tf.get_variable_scope().reuse_variables()
        
        
        avg_grads = average_gradients(tower_grads)
        minimize_op = optimizer.apply_gradients(avg_grads, global_step=step)

        ema_op = exp_mov_avg.apply([v for v  in tf.trainable_variables()])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #train_op = tf.group(minimize_op, update_ops)
        train_op = tf.group(minimize_op, update_ops, ema_op)


        pred_classes_array = tf.concat(values=pred_classes_array,axis=0)
        pred_boxes_array = tf.concat(values=pred_boxes_array,axis=0)
        
        

        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        var_load_fn = get_var_load_fn(last_model, pretrained_feature_path)
        
        

    #pdb.set_trace()
    with tf.Session(graph=graph) as sess:
        print('Start counting time ..')
        
        sess.run(global_init)
        var_load_fn(sess)

        start_epoch = 0
        if last_model is not None:
            print('last step = ', last_step)
            trn_iter = sess.run(tf.assign(step,last_step+1))
            last_epoch = int(last_step/nu_epoch0)
            start_epoch = last_epoch
        elif pretrained_feature_path is not None:
            trn_iter=0

        nbatches = int(nsamples/batch_size_total)
        start_time = time.time() 
        min_npos = float(1e6)
        for ep in range(start_epoch, args.num_epochs):
            for bb in range(nbatches):
                #pdb.set_trace()
                
                if (trn_iter%250) ==0:
                    closs,bloss,lloss,lr,_ = sess.run([class_loss_array,box_loss_array,l2_loss_array, learning_rate,train_op])
                    elapsed_time = time.time() - start_time
                    print('iter= ', trn_iter, 'lr= ',lr,'cls= ',np.amax(closs), 'bls= ', np.amax(bloss), 'l2= ',np.amax(lloss), 'time = '+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), 'minN=',min_npos )
                    start_time = time.time()

                else:
                    npos,nneg,trn_fnames, _ = sess.run([nmatches, nnegs,filename, train_op])
                    min_npos = min(npos, min_npos)
                    #print([v[0].decode('ascii').split('/')[-1] for v in trn_fnames])

                trn_iter+=1

                if (trn_iter % 5000) == 0:
                    saver0 = tf.train.Saver()
                    modelName = 'model-'+str(trn_iter).zfill(6)
                    modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
                    saver0.save(sess, modelPath)
                    saver0.export_meta_graph(modelPath+'.meta')
           
if __name__=='__main__':
    args = parser.parse_args()
    args.dataset_meta=args.dataset_meta.split(',')
    args.exclude_img_list=args.exclude_img_list.split(',')
    main(args)
