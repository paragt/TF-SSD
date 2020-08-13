import os,sys
import argparse
import pdb
import tensorflow as tf
import numpy as np
import json
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/..')

from my_config import *
import ssd_augmenter_vgg
import ssd300, ssd_common
from draw_detections import mark_detections
from multigpu import average_gradients
import losses
import preprocess
import dataset_feeder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/tf_objdet/trn_tf_script/data_COCO/')
parser.add_argument('--dataset_meta', default='train2014')
parser.add_argument('--exclude_img_list', default='none')
parser.add_argument('--category_map_file', default='none')
parser.add_argument('--mode', default='train')#'train')
parser.add_argument('--resize_method', default='BILINEAR', choices=['BILINEAR', 'ASPECT', 'PAD', 'NONE'])
parser.add_argument('--lr_steps',    default=    "1000, 2000,3000 , 4000 ,  100000, 200000")
parser.add_argument('--lr_values', default="0.00005, 0.0001,0.0003,0.0007,0.001,0.0001,0.00001")
parser.add_argument('--smallest_ratio', default="4,10")
parser.add_argument('--resolution', default=300, type=int)
parser.add_argument('--num_epochs', default=3, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--pretrained_feature_path', default='', type=str)
parser.add_argument('--confidence_threshold', default=0.3, type=float)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float) # google  uses 4e-5, was using 5e-4 before
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--error_normalize_factor', default=1.0, type=float)
parser.add_argument('--huber_loss_delta', default=1.0, type=float)






def get_lr_optimizer(step, boundaries, values):

    #boundaries = [2*nu_epoch, 3*nu_epoch, 100*nu_epoch, 150*nu_epoch]
    #values = [0.00005,     0.0001,   init_lr,      init_lr*0.1,  init_lr*0.05]
    #boundaries = [2*nu_epoch, 4*nu_epoch, 6*nu_epoch, 8*nu_epoch,  100*nu_epoch, 150*nu_epoch]
    #values = [0.00005,     0.0001,    0.0003,     0.0007,       init_lr,      init_lr*0.1,  init_lr*0.05]
    #boundaries = lr_boundaries
    #values = lr_values
        
    learning_rate_piecewise = tf.train.piecewise_constant(step, boundaries, values)

    #decay_step_exp = nu_epoch*2
    #learning_rate_exp = tf.train.exponential_decay(init_lr, step, decay_step_exp, 0.94, staircase=True)

    learning_rate = learning_rate_piecewise
    tf.summary.scalar('lr',learning_rate)

    #tf.summary.scalar('epsilon',epsilon)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=False)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,epsilon=1.0)
    #optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)

    return learning_rate, optimizer


def get_var_load_fn(last_model, pretrained_feature_path):
    if last_model is not None:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(last_model,  variables_to_restore)
    elif pretrained_feature_path is not None:
        variables_to_restore = [v for v  in tf.trainable_variables() if 'vgg_16' in v.name]
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_feature_path,  variables_to_restore)

    #other_variables = [v for v in tf.global_variables() if (v not in variables_to_restore and 'batch_normalization' in v.name and 'moving' in v.name)]
    #variables_to_restore.extend(other_variables)
    return var_load_fn


def main(args):

    pdb.set_trace()

    lr_boundaries = [int(x) for x in args.lr_steps.split(',')]
    lr_values = [float(x) for x in args.lr_values.split(',')]


    smallest_ratio = [int(x) for x in args.smallest_ratio.split(',')]
    ANCHORS_MAP, NUM_ANCHORS = ssd_common.get_anchors(ANCHORS_STRIDE,ANCHORS_ASPECT_RATIOS,MIN_SIZE_RATIO, MAX_SIZE_RATIO, INPUT_DIM, smallest_ratio)

    resize_method = args.resize_method
    #mscoco = dataset_feeder.MscocoFeeder(args,ssd_augmenter_vgg.augment, resize_method)
    #mscoco = dataset_feeder.MscocoFeeder(args,ssd_augmenter_vgg.augment)

    dtst_list=[]
    for ii,dtst_name in enumerate(args.dataset_meta):
        mscoco1 = dataset_feeder.MscocoFeeder(args, dtst_name, exclude_img_list_file=args.exclude_img_list[ii], category_map_file=args.category_map_file)
        dtst_list.append(mscoco1)


    nsamples = sum([v.num_samples for v in dtst_list])
    nclasses = len(dtst_list[0].cat_names)+ 1  
    batch_size_per_gpu = args.batch_size
    batch_size_total = args.num_gpus*batch_size_per_gpu
    pretrained_feature_path = args.pretrained_feature_path 
    exm_per_epoch = nsamples
    modelDir = args.model_dir

    tfobj = preprocess.TFdata(args, nsamples, ssd_augmenter_vgg.augment)

    last_model, last_step = check_for_existing_model(modelDir)
    nu_epoch0 = exm_per_epoch*1.0/(batch_size_per_gpu*args.num_gpus)


    #get_sample = mscoco.get_samples_fn()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        step=tf.train.get_or_create_global_step()
        nu_epoch = tf.cast(nu_epoch0, tf.int32)
        #nu_epoch = tf.cast(exm_per_epoch*1.0/(batch_size_per_gpu*args.num_gpus), tf.int32) # num_update_per_epoch
        init_lr = args.learning_rate

        boundaries = lr_boundaries
        values = lr_values
        learning_rate, optimizer = get_lr_optimizer(step, boundaries, values)



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
                    pred_classes, pred_boxes  = ssd300.net(split_images[i], nclasses, NUM_ANCHORS, True, 'my_vgg3')

                    loss, class_loss, bboxes_loss, loss_l2, nmatches, nnegs = losses.compute_loss(pred_classes, pred_boxes, split_classes[i], split_boxes[i], split_mask[i], huber_loss_delta=args.huber_loss_delta, error_normalize_factor=args.error_normalize_factor, weight_decay=args.weight_decay)


                    grads = optimizer.compute_gradients(loss) #, var_list=self.train_vars)

                    tower_grads.append(grads)

                    pred_classes_array.append(pred_classes)
                    pred_boxes_array.append(pred_boxes)
                    class_loss_array.append(class_loss)
                    box_loss_array.append(bboxes_loss)
                    l2_loss_array.append(loss_l2)
                    
                    tf.get_variable_scope().reuse_variables()
        
        
        avg_grads = average_gradients(tower_grads)
        minimize_op = optimizer.apply_gradients(avg_grads, global_step=step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)


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

        #classes , boxes, split_classes, split_boxes = sess.run([classes,boxes,split_classes,split_boxes])
        nbatches = int(nsamples/batch_size_total)
        #trn_iter = 0
        start_time = time.time() 
        for ep in range(start_epoch, args.num_epochs):
            for bb in range(nbatches):
                
                if (trn_iter%250) ==0:
                    closs,bloss,lloss,lr,_ = sess.run([class_loss_array,box_loss_array,l2_loss_array, learning_rate,train_op])
                    elapsed_time = time.time() - start_time
                    #print('time needed = '+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                    print('iter= ', trn_iter, 'lr= ',lr,'cls= ',np.amax(closs), 'bls= ', np.amax(bloss), 'l2= ',np.amax(lloss), 'time = '+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    start_time = time.time()
                else:
                    npos,nneg,_ = sess.run([nmatches, nnegs, train_op])

                trn_iter+=1

                if (trn_iter % 5000) == 0:
                    saver0 = tf.train.Saver()
                    modelName = 'model-'+str(trn_iter).zfill(6)
                    modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
                    saver0.save(sess, modelPath)
                    saver0.export_meta_graph(modelPath+'.meta')
           
        #output_name = 'detections_'+args.dataset_meta[0]+'.json'
        #with open(output_name,'wt') as fid:
            #json.dump(all_det, fid)
        
if __name__=='__main__':
    args = parser.parse_args()
    args.dataset_meta=args.dataset_meta.split(',')
    main(args)




'''
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--epoch_size', default=0, type=int)
parser.add_argument('--cost_balance', default=10., type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)


def create_tfdataset(feeder, batch_size, num_epochs):

    dataset = tf.data.Dataset.from_generator(generator=lambda: feeder.get_samples_fn(), output_types=(tf.int64, tf.string, tf.int64, tf.float32))

    dataset = dataset.shuffle(feeder.get_num_samples())

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(lambda image_id, file_name, classes, boxes: feeder.parse_fn(image_id, file_name, classes, boxes), num_parallel_calls=12)

    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None, None, 3], [None], [None, 4], [None], [None], [None])) # id, image, classes, boxes, scale, translation

    dataset = dataset.prefetch(2)
    return dataset


def check_for_existing_model(modelDir):

    last_model = None
    last_epoch = -1

    if not os.path.exists(modelDir):
        return last_model, last_epoch
    filelist = os.listdir(modelDir)

    #pdb.set_trace()
    metafiles = [x for x in filelist if '.meta' in x]
    if len(metafiles)<1:
        print('no existing network')
        return last_model, last_epoch
    else:
        for mf in metafiles:
            mf_noextn = os.path.splitext(mf)[0]
            mf_split = mf_noextn.split('-')
            epoch = int(mf_split[1])
            if epoch > last_epoch:
                last_epoch = epoch
                last_model = os.path.join(modelDir,mf_noextn)

        return last_model, last_epoch





                    gt_mask = tf.reshape(split_mask[i], [-1])
                    logits_classes = tf.reshape(pred_classes, [-1, tf.shape(pred_classes)[2]])
                    gt_classes = tf.reshape(split_classes[i], [-1, 1])
                    logits_bboxes = tf.reshape(pred_boxes, [-1, 4])
                    gt_bboxes = tf.reshape(split_boxes[i], [-1, 4])
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

                     
                    #fg_cls_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=fg_labels, logits=fg_logits))
                    #bg_cls_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=bg_labels, logits=bg_logits))
                    fg_cls_loss = tf.losses.sparse_softmax_cross_entropy(labels=fg_labels, logits=fg_logits,reduction=tf.losses.Reduction.SUM)
                    bg_cls_loss = tf.losses.sparse_softmax_cross_entropy(labels=bg_labels, logits=bg_logits,reduction=tf.losses.Reduction.SUM)

                    pred_bboxes_select = tf.gather(logits_bboxes, fg_index)
                    gt_bboxes_select = tf.gather(gt_bboxes, fg_index)

                    pred_bboxes_select = tf.reshape(pred_bboxes_select,[-1,4])                   
                    gt_bboxes_select = tf.reshape(gt_bboxes_select,[-1,4])                   
 
                    bboxes_loss = tf.losses.huber_loss(gt_bboxes_select, pred_bboxes_select)

                    nmatches = tf.cast(tf.shape(fg_logits)[0],tf.float32)
                    nmatches = tf.clip_by_value(nmatches, tf.cast(batch_size_per_gpu, tf.float32), nmatches)
                    nnegs = tf.cast(tf.shape(bg_logits)[0],tf.float32)
                    class_loss = (fg_cls_loss + bg_cls_loss)/nmatches
                    
                    
                    #class_losses, bboxes_losses = ssd_common.loss([split_classes[i], split_boxes[i], split_mask[i]], [pred_classes, pred_boxes], CLASS_WEIGHT, BOX_WEIGHT)
                    

                    l2_var_list = [v for v in tf.trainable_variables() if not any(x in v.name for x in SKIP_L2_LOSS_VARS)]

                    loss_l2 = args.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in l2_var_list])

                    loss = tf.identity(class_loss+ bboxes_loss + loss_l2, "total_loss")

'''
