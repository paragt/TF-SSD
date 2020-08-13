
# Priorboxes
ANCHORS_STRIDE = [8, 16, 32, 64, 100, 300]
#ANCHORS_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

ANCHORS_ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
MIN_SIZE_RATIO = 15 # 0.15 or  0.2
MAX_SIZE_RATIO = 90 # 0.9 or 0.95

#ANCHORS_ASPECT_RATIOS = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
#MIN_SIZE_RATIO = 15 # 0.15 or  0.2
#MAX_SIZE_RATIO = 90 # 0.9 or 0.95

INPUT_DIM = 300

CLASS_WEIGHT = 1.0
BOX_WEIGHT = 1.0

SKIP_L2_LOSS_VARS=['l2_norm_scaler', 'BatchNorm']


# control the size of the default square priorboxes
# REF: https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp#L164

import os

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

