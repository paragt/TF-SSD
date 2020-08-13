#CUDA_VISIBLE_DEVICES=6,7 python my_train.py --num_gpus=2 --num_epochs=700 --pretrained_feature_path=../../../lambda-deep-learning-demo/models/pretrained/VGG_16_reduce/model-tensorflow-renamed --model_dir=models/vgg_reduced_ssd_alldet_bilinear_smallest4


CUDA_VISIBLE_DEVICES=6,7 python my_train.py --num_gpus=2 --num_epochs=700 --pretrained_feature_path=../../../lambda-deep-learning-demo/models/pretrained/VGG_16_reduce/model-tensorflow-renamed --model_dir=models/vgg_reduced_ssd_alldet_bilinear_smallest4_lr03 --lr_steps=1000,2000,3000,4000,70000,140000,210000,280000,350000  --lr_values=0.00005,0.0001,0.0005,0.001,0.003,0.001,0.0003,0.0001,0.00003,0.00001
