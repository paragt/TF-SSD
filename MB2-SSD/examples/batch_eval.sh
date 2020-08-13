#!/bin/bash

GPU=$1
modeldir=$2

#for filename in $(find $modeldir  -name "*.index" ); 
for idx in 500000 525000 550000 575000 600000 625000 650000 675000 700000
#for idx in 630000 
do 
    #echo $i; 
    #extension="${filename##*.}"
    #nameonly="${filename%.*}"
    echo ${modeldir}/model-${idx}   
    CUDA_VISIBLE_DEVICES=$GPU python3 my_eval_mb.py --batch_size=1 --confidence_threshold=0.01 --minival_ids_file=../../frozen/mscoco_minival2014_ids.txt --pretrained_model_path=${modeldir}/model-${idx}
done


