
# # for train2014

CUDA_VISIBLE_DEVICES=4,5 python my_train_mb.py --num_gpus=2 --pretrained_feature_path=../pretrained/mobilenet/mobilenet_v2_1.4_224.ckpt --batch_size=32 --dataset_meta=train2014 --model_dir=models/mb2ssd_train2014/ --lr_steps=1000,2000,70000,140000,210000,280000,350000,420000 --lr_values=0.00001,0.0001,0.009,0.003,0.0009,0.0003,0.00009,0.00003,0.00001

# # for {train2014 U val2014} \ minival2014

CUDA_VISIBLE_DEVICES=4,5 python my_train_mb.py --num_gpus=2 --pretrained_feature_path=../pretrained/mobilenet/mobilenet_v2_1.4_224.ckpt --batch_size=32 --dataset_meta=train2014,val2014 --exclude_img_list=None,/home/tf_objdet/trn_tf_script/data_COCO/mscoco_minival2014_ids.txt --model_dir=models/mb2ssd_trainval2014_3/ --lr_steps=1000,2000,100000,200000,300000,400000,500000,600000 --lr_values=0.00001,0.0001,0.01,0.0033,0.001,0.00033,0.0001,0.000033,0.00001
