import json
import os,sys
import cv2
from PIL import Image
import numpy as np
import pdb
import shutil

def mark_detections(input_im, classes, boxes, imgpath):


    #pdb.set_trace()
    oimg = input_im.astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    imh,imw = input_im.shape[:2]
    for ii,cls in enumerate(classes):
        #cls = obj['Object']
        box =  boxes[ii] #obj['box']
        xmin = int(box[0]*imw)
        xmax = int(box[2]*imw)
        ymin = int(box[1]*imh)
        ymax = int(box[3]*imh)

        oimg = cv2.rectangle(oimg, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
        oimg=cv2.putText(oimg, str(cls) , (xmin, ymin+4), font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


    saveim = Image.fromarray(oimg)
    savepath = os.path.splitext(imgpath)[0]+ '_detections.jpg'
    saveim.save(savepath)


if __name__ == '__main__':

    #pdb.set_trace()
    jsonname = sys.argv[1]
    foldername = jsonname.replace('.json','')
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

    with open(jsonname) as fid:
        data = json.load(fid)

    for imname in data:
        im = np.array(Image.open(imname))
        cls = np.array(data[imname]['class_name'])
        scores = np.array(data[imname]['score'])
        boxes = np.array(data[imname]['box'])

        basename = os.path.basename(imname)
        print(basename, 'shape ', im.shape, 'ndet = ', len(boxes))
        mark_detections(im, cls, boxes, os.path.join(foldername,basename))
