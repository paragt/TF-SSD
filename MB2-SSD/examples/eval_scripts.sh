
#on val2014
CUDA_VISIBLE_DEVICES=0 python my_eval_mb_ema.py --batch_size=1 --confidence_threshold=0.01 --pretrained_model_path=../../models/model_mb_d1.4_ssd_ndet_allerr_bilinear_smallest4_aspect6_bn_decay1e-4_ema0.9999_adam/model-400000



# # on minival2014

# # tf group  version

CUDA_VISIBLE_DEVICES=0 python my_eval_frozen.py  --confidence_threshold=0.01  --model_path=frozen/ssd_mobilenet_v2_coco_2018_03_29/frozen_thd0.01/frozen_inference_graph.pb --labelmap_path=frozen/mscoco_label_map.pbtxt


# # our version

CUDA_VISIBLE_DEVICES=0 python3 my_eval_mb.py --batch_size=1 --confidence_threshold=0.01 --minival_ids_file=../../frozen/mscoco_minival2014_ids.txt --pretrained_model_path=models/mb2ssd_trainval2014_3/model-675000


CUDA_VISIBLE_DEVICES=0 python3 my_eval_mb.py --batch_size=1 --confidence_threshold=0.01 --minival_ids_file=../../frozen/mscoco_minival2014_ids.txt --pretrained_model_path=models/mb2ssd_trainval2014_3/model-700000
