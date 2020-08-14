# TF-SSD
### A Simple Tensorflow Implementation of Single Shot Detector (SSD) trained on MSCOCO 2014 dataset.

This repo contains codes, scripts for training and applying SSDs [Liu16ECCV] for general object detection. The codes were written in Tensorflow version 1.12. The current implementation trains the detectors on standard [MSCOCO 14 dataset](https://cocodataset.org/#home) and evaluated on MSCOCO val 2014 set or its subset **and** on test-dev 2017. 

The implementation of modeling, training and application of the detectors was intended to be as simple and modular as possible. The code attempts to avoid additional wrappers/abstractions (e.g., Estimators) as well as complex organizations so that the core functionalities of the actual CNN architecture is readily visible and can easily be resturctured for any other configurations. I believe, understanding the fundamental details of SSD is essential to make a significant contribution to for object detection in general.

Despite its lean nature, it attains, in fact exceeds, the performances of the original implementations (details provided below).

This codebase was built upon the [github repo](https://github.com/lambdal/lambda-deep-learning-demo) from Lambda AI  (by Chuan Li) which itself closely follows the [Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd/examples/ssd) from the author of SSD paper. Some parts of the code is taken directly from the Lambda AI repo. However, in addition to simplifying the code, I have improved the accuracy by revising the training techniques and have extended it to include another popular backbone (MobileNetV2) for the SSD. I wish to also cknowledge the [Tensorflow research group](https://github.com/tensorflow/models/tree/master/research) for making their implementation and trained models public.

## SSD with VGG-16 backbone.

The first and original version of SSDs used VGG networks as their backbones [Liu16ECCV]. We use VGG-16 as the backbone and the code for this variant are saved in the VGG-SSD subfolder. The function of each file is self evident from its name, e.g., my_vgg3.py defines the backbone, ssd300.py defines the SSD connections, ssd_augmenter_vgg.py performs augmentation, my_train.py runs the training scheme with losses defined in losses.py. Some common operations, such as defining default/prior/anchor boxes, prediction classifiers etc. and dataset parsing are contained in ssd_common.py and dataset_feeder.py in the parent folder. 

The current VGG-SSD model uses an input of 300x300 pixels and predicts 80 categories as specified by the COCO dataset. Other than the size of the default boxes at a certain scale and data augmentation techniques, the model and training remains almost the same as those of the SSD paper. 

Following standard protocol, the accuracies of the SSD trained on COCO are reported in terms of the mAP @[0.5:0.95] IoU. In the examples subfolder, we provide the exact command used to train and evaluate the detectors. The example folder also contains results from the [COCO 2019 evaluation server](https://competitions.codalab.org/competitions/20794).  The zip file we submitted to COCO 2019 server can be found [here](https://drive.google.com/file/d/17tIEcc9kxyGEOp3uLaPqOTaEvKJA2Si1/view?usp=sharing).

| Implementation | Trn Set | mAP COCO14 Val | mAP COCO17 test-dev|
| :--- | :---: | :---: | :---: |
|[LambdaAI](https://drive.google.com/file/d/1xp7B3WHudEkDjcVSVAaRSa8umQIad69b/view?usp=sharing) | COCO14 Trn | 22.7 | see * below |
|[Ours](https://drive.google.com/file/d/1B-cb7b_3UfEu_HFmlBr1-Uw5EH8m4nOu/view?usp=sharing) | COCO14 Trn| **24.0** | **23.4** |

\* According to their [blog](https://lambdalabs.com/blog/how-to-implement-ssd-object-detection-in-tensorflow/) and [demo site](https://lambda-deep-learning-demo.readthedocs.io/en/latest/tutorial/ssd.html#evaluate-ssd-on-mscoco), VGG-SSD300 of LambdaAI implementation can achieve mAP of 21.9 on COCO 17 Val set.

The pretrained VGG net for feature backbone converted to Tensorflow format can be found [here](https://drive.google.com/file/d/1mnJSilb5vfi3yc6bD_cifLSQAEBXEtiI/view?usp=sharing).
 
This implementation uses 12 as the smallest scale (as opposed to 21) of the default (prior/anchor) boxes at stride 16 output of VGG (i.e., conv4_3). We also do not use color distortion for data augmentation, resorting only to ssd_random_crop and random horizontal flips. 

During training, the optimizer minimizes the loss, as defined by [Liu16ECCV], exactly. In order to cope with the initial large classification loss values, due to the normalization by number of deault boxes (*NOT by averaging*), we use a simple warmup learning rate scheme. See train_scripts.sh in examples subfolder for details. We also played with the learning rate schedule a little bit to improve accuracy. 




## SSD with Mobilenet V2 backbone

A later study by [] adopted Mobilenets as feature generators for efficiency. Our implementation utilizes Mobilenet V2 []  and the codebase for this variant is organized in a similar fashion within MB2-SSD subfolder. Likewise, the functionality of the codes are evident from their names. 

The input to the detector is 300x300, but can be modified as needed. As we are using the pretrained model from the Tensorflow [model depot](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md), the data preprocessing (so called Inception style) is different from VGG-SSD. However, as we observed, it is not as straightforward as plugging in a MobileNetV2 model in front of an SSD to attain the already published accuracy. We list the modifications needed to not just achieve, but to exceed, the accuracy (in mAP) of its counterpart from Tensorflow [Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) on COCO test-dev17 dataset evaluated by CoLab.

As before we are reporting results in terms of mAP@[0.5:0.95] IoU on different datasets and subsets. The script to generate this values are provided in the examples subfolder.


| Implementation | Trn Set | mAP COCO14 miniVal14|  mAP COCO17 test-dev | Inf time(msec) |
| :--- | :---: | :---: | :---: | :---: |
| [TF group](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | Trn 14, Val 14 \ miniVal14 (?) | 21.5 | 20.7 | 46 |
| [Ours 1](https://drive.google.com/file/d/1HwNKbww72_R7kGxRu2Yody2n2oAQn53A/view?usp=sharing) | Trn 14, Val 14 \ miniVal14 | 21.9 | 21.2 | 28 |

Following Tensorflow [Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), we used the same [image ids file](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt) to create miniVal14 subset. One has to configure the frozen inference graph of TF group so that it predicts above a threshold of 0.01 for fair comparison. We have done this and saved the model and necessary scripts [here](https://drive.google.com/file/d/1GZBAioueHyCKTBTSrvWUco2pvEuKGHW1/view?usp=sharing). Our submission to COCO 2019 can also be found [here](https://drive.google.com/file/d/1u3gfae3HLvMn3YRrQrMnwmJiTOrMU_05/view?usp=sharing).

The inference time for one image was calculated by averaging the time required to predict the COCO14 miniVal14 set on the same computer with Nvidia GTX 1080 gpus. We must point out that the inference time depends heavily on the overall configuration of the hardware and may be different on other machines -- in this table, we use the inference time to compare the speed of the two versions (TF group and ours). Check the my_eval_mb.py in MB2-SSD folder and my_eval_frozen.py in [here](https://drive.google.com/file/d/1GZBAioueHyCKTBTSrvWUco2pvEuKGHW1/view?usp=sharing) to understand how the runtime was computed. 

Based on our analysis, we speculate that the MB2-SSD from Tensorflow group (TF group) was trained on COCO Trn14 + Val14 - miniVal14 images  -- we added a ? mark since we are not 100% sure of this. We have also trained a model on only COCO Trn14, the accuracy of this model is as follows.

| Implementation | Trn Set | mAP COCO14 Val |
| :--- | :---: | :---: | 
| [Ours 2](https://drive.google.com/file/d/1uUOb9BpOmTOwrUEfqasr5pkO36V7_uuT/view?usp=sharing) | Trn 14 | 20.0 |

The pretrained backbone for both our models is saved [here](https://drive.google.com/file/d/1vfZG4JhOEeQnNlf-41QpbZt8kWLzpOWZ/view?usp=sharing). In the following text, we discuss the modifications to the architecture and the training that led to the aforementioned accuracy.




[[Liu16ECCV](https://arxiv.org/abs/1512.02325)] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott E. Reed, Cheng-Yang Fu and Alexander C. Berg (2016). ECCV 2016.
