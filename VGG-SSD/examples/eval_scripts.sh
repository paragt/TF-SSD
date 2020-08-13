

# orig lambdaAI implementation

CUDA_VISIBLE_DEVICES=0 python demo/image/object_detection.py --mode=eval --model_dir=models/ssd300_mscoco/model.ckpt-128000  --network=ssd300 --augmenter=ssd_augmenter --batch_size_per_gpu=4 --epochs=1 --dataset_dir=/home/tf_objdet/trn_tf_script/data_COCO/ --num_classes=81 --resolution=300 --confidence_threshold=0.01 --feature_net=vgg_16_reduced --feature_net_path=models/pretrained/VGG_16_reduce/VGG_16_reduce.p eval_args --dataset_meta=val2014 --reduce_ops=False --callbacks=eval_basic,eval_speed,eval_mscoco


# # this implementation


#CUDA_VISIBLE_DEVICES=4 python my_eval.py --batch_size=1 --confidence_threshold=0.01 --pretrained_model_path=models/vgg_reduced_ssd_alldet_bilinear_smallest4/model-410000

CUDA_VISIBLE_DEVICES=0 python my_eval.py --batch_size=4 --confidence_threshold=0.01 --pretrained_model_path=models/vgg_reduced_ssd_alldet_bilinear_smallest4_lr03/model-305000 
