# TF-SSD
### A Simple Tensorflow Implementation of Single Shot Detector (SSD) trained on MSCOCO 2014 dataset.

This repo contains codes, scripts for training and applying SSDs for general object detection. The codes were written in Tensorflow version 1.12. The current implementation trains the detectors on standard MSCOCO 14 dataset and evaluated on MSCOCO val 2014 set or its subset **and** on test-dev 2017. 

The implementation of modeling, training and application of the detectors was intended to be as simple and modular as possible. The code attempts to avoid additional wrappers/abstractions (e.g., Estimators) as well as complex organizations so that the core functionalities of the actual CNN architecture is readily visible and can easily be resturctured for any other configurations. I believe, understanding the fundamental details of SSD is essential to work on improvements or invent novel methods for object detections in general.

Despite its lean nature, it attains, in fact exceeds, the performances of the original implementations (details provided below).

This codebase was built upon the [github repo](https://github.com/lambdal/lambda-deep-learning-demo) from Lambda AI  (by Chuan Li) which itself closely follows the [Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd/examples/ssd) from the author of SSD paper. Some parts of the code is taken directly from the Lambda AI repo. However, in addition to simplifying the code, I have improved the accuracy by revising the training techniques and have extended it to include another popular backbone (MobileNetV2) for the SSD. I wish to also cknowledge the [Tensorflow research group](https://github.com/tensorflow/models/tree/master/research) for making their implementation and trained models public.

## SSD with VGG-16 backbone.
The code for this variant are saved in the VGG-SSD subfolder. The function of each file is self evident from its name, e.g., my_vgg3.py defines the backbone, ssd300.py defines the SSD connections, ssd_augmenter_vgg.py performs augmentation, my_train.py runs the training scheme with losses defined in losses.py. Some common operations, such as defining default/prior/anchor boxes, prediction classifiers etc. and dataset parsing are contained in ssd_common.py and dataset_feeder.py in the parent folder. 

The current VGG-SSD model uses an input of 300x300 pixels and predicts 80 categories as specified by the COCO dataset. Other than the size of the default boxes at a certain scale and data augmentation techniques, the model and training remains almost the same as those of the SSD paper. 

Following standard protocol, the accuracies of the SSD trained on COCO are reported in terms of the mAP @[0.5:0.95] IoU. In the examples subfolder, we provide the exact command used to train and evaluate the detectors. The example folder also contains results from the [COCO 2019 evaluation server](https://competitions.codalab.org/competitions/20794)

| Implementation | Trn Set | mAP COCO14 Val | mAP COCO17 test-dev|
| :--- | :---: | :---: | :---: |
|[LambdaAI](https://drive.google.com/file/d/1xp7B3WHudEkDjcVSVAaRSa8umQIad69b/view?usp=sharing) | COCO14 Trn | 22.7 | see below |
|Ours | COCO14 Trn| **24.0** | **23.4** |


