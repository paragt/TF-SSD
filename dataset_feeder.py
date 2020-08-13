from __future__ import print_function
import os
import numpy as np
import math
import pdb
import tensorflow as tf

from pycocotools.coco import COCO


JSON_TO_IMAGE = {
    "train2017": "train2017",
    "val2017": "val2017",
    "train2014": "train2014",
    "val2014": "val2014",
    "valminusminival2014": "val2014",
    "minival2014": "val2014",
    "test2014": "test2014",
    "test2017": "test2017",
    "test-dev2015": "val2017"
}


class MscocoFeeder():
  def __init__(self, config, dataset_meta, exclude_img_list_file='None', category_map_file='None', use_bckgnd=False):

    self.config = config
    self.dataset_meta = dataset_meta
    self.category_id_to_class_id = None
    self.class_id_to_category_id = None
    self.cat_names = None
    self.exclude_img_list = []
    self.use_bckgnd = use_bckgnd
    self.category_map_file = category_map_file

    if os.path.exists(exclude_img_list_file):
    #if exclude_img_list_file.lower() != 'none':
        self.exclude_img_list = np.loadtxt(exclude_img_list_file).astype(np.int32).tolist()
    else:
        print('exclude img list file ', exclude_img_list_file, ' does not exist')

    if self.config.mode == "infer":
      self.test_samples = self.config.test_samples
    elif self.config.mode == "export":
      pass
    else:
      self.parse_coco()

    self.num_samples = self.get_num_samples()

  def rearrange_categories(self):
    with open(self.category_map_file,'rt') as fid:
      cat_data = fid.read()

    cat_data=cat_data.split('\n')[:-1]
    cat_data=[v for v in cat_data if v.strip()[0]!='#']
    override_name_to_catid = {}
    override_catid_to_name = {}
    override_category_id_to_class_id = {}
    override_class_id_to_category_id = {}
    for line1 in cat_data:
      cname = line1.split(':')[0].strip()
      class_id = int(line1.split(':')[1].strip())
      if cname in self.name_to_catid.keys():
        coco_id = self.name_to_catid[cname]

        override_name_to_catid[cname] = coco_id
        override_catid_to_name[coco_id] = cname

        override_category_id_to_class_id[coco_id] = class_id
        override_class_id_to_category_id[class_id] = coco_id 

     
    self.name_to_catid = override_name_to_catid
    self.catid_to_name = override_catid_to_name
 
    self.category_id_to_class_id = override_category_id_to_class_id
    self.class_id_to_category_id = override_class_id_to_category_id

    self.cat_names = [v for v in self.name_to_catid.keys()]

  def parse_coco(self):
    samples = []
    #for name_meta in self.dataset_meta:
    name_meta = self.dataset_meta


    if True:
      annotation_file = os.path.join(
        self.config.dataset_dir,
        "annotations",
        "instances_" + name_meta + ".json")

      if not os.path.exists(annotation_file):
        annotation_file = os.path.join(self.config.dataset_dir,"annotations","image_info_" + name_meta + ".json")
      coco = COCO(annotation_file)

      #pdb.set_trace()
      cat_ids = coco.getCatIds()
      self.cat_names = [c["name"] for c in coco.loadCats(cat_ids)]
      self.catid_to_name = { rec['id']:rec['name'] for rec in coco.loadCats(cat_ids) }      
      self.name_to_catid = { rec['name']:rec['id'] for rec in coco.loadCats(cat_ids) }   
      class_id_list = [v for v in range(1,len(cat_ids)+1)]# background has class id of 0
      self.category_id_to_class_id = {
        v: i  for i, v in zip(class_id_list, cat_ids)}
      self.class_id_to_category_id = {
        v: k for k, v in self.category_id_to_class_id.items()}


      if os.path.exists(self.category_map_file):
        self.rearrange_categories()
    

      img_ids = coco.getImgIds()
      img_ids.sort()

      # list of dict, each has keys: height,width,id,file_name
      imgs = coco.loadImgs(img_ids)

      for img in imgs:
        img["file_name"] = os.path.join(
          self.config.dataset_dir,
          JSON_TO_IMAGE[name_meta],
          img["file_name"])

      if self.config.mode == "train" :
        for img in imgs:
          self.parse_gt(coco, self.category_id_to_class_id, img)
        #imgs2=[v for v in imgs if np.sum(v['is_crowd'])<1] # exclude images with 'is_crowd'
        #imgs = imgs2

      samples.extend(imgs)

    if len(self.exclude_img_list)>0 :
      samples = list(filter(lambda sample: sample['id'] not in self.exclude_img_list, samples))

    # Filter out images that has no object.
    if self.config.mode == "train" and (not self.use_bckgnd):
      samples = list(filter(
        lambda sample: len(
          sample['boxes'][sample['is_crowd'] == 0]) > 0, samples))

 
    self.samples = samples


  
  def get_num_samples(self):
    return len(self.samples)
    '''
    if not hasattr(self, 'num_samples'):
      if self.config.mode == "infer":
        self.num_samples = len(self.test_samples)
      elif self.config.mode == "export":
        self.num_samples = 1        
      elif self.config.mode == "eval":
        self.num_samples = 0 #self.EVAL_NUM_SAMPLES
        for name_meta in self.config.dataset_meta:
          subdir_path = os.path.join(self.config.dataset_dir, name_meta)
          self.num_samples += len(os.listdir(subdir_path))
      elif self.config.mode == "train":
        self.num_samples = 0 #self.TRAIN_NUM_SAMPLES
        for name_meta in self.config.dataset_meta:
          subdir_path = os.path.join(self.config.dataset_dir, name_meta)
          self.num_samples += len(os.listdir(subdir_path))
    return self.num_samples
    '''
  

  def parse_gt(self, coco, category_id_to_class_id, img):
    ann_ids = coco.getAnnIds(imgIds=img["id"], iscrowd=None)
    objs = coco.loadAnns(ann_ids)

    # clean-up boxes
    valid_objs = []
    width = img["width"]
    height = img["height"]

    for obj in objs:
      if obj['category_id'] not in category_id_to_class_id:
        continue
      #pdb.set_trace()
      if obj.get("ignore", 0) == 1:
        print('ignore found in obj')
        continue
      x1, y1, w, h = obj["bbox"]

      x1 = float(x1)
      y1 = float(y1)
      x2 = float(x1 + w)
      y2 = float(y1 + h)

      x1 = max(0, min(float(x1), width - 1))
      y1 = max(0, min(float(y1), height - 1))
      x2 = max(0, min(float(x2), width - 1))
      y2 = max(0, min(float(y2), height - 1))

      w = x2 - x1
      h = y2 - y1

      if obj['area'] > 1 and w > 0 and h > 0 and w * h >= 4:
        # normalize box to [0, 1]
        obj['bbox'] = [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]
        valid_objs.append(obj)

    boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)

    cls = np.asarray([
        category_id_to_class_id[obj['category_id']]
        for obj in valid_objs], dtype='int32')  # (n,)

    is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

    img['boxes'] = boxes # nx4
    img['class'] = cls # n, always >0
    img['is_crowd'] = is_crowd # n,


  

  def get_samples_fn(self):
    # Args:
    # Returns:
    #     sample["id"]: int64, image id
    #     sample["file_name"]: , string, path to image
    #     sample["class"]: (...,), int64
    #     sample["boxes"]: (..., 4), float32
    # Read image
    if self.config.mode == "infer":
      for file_name in self.test_samples:
        yield (0,
               file_name,
               np.empty([1], dtype=np.int32),
               np.empty([1, 4]))
    elif self.config.mode == "eval":
      for sample in self.samples[0:self.num_samples]:
        yield(sample["id"],
              sample["file_name"],
              np.empty([1], dtype=np.int32),
              np.empty([1, 4]))
    else:
      for sample in self.samples[0:self.num_samples]:
        if sample['boxes'].shape[0] > 0:
          # remove crowd objects
          mask = sample['is_crowd'] == 0
          sample["class"] = sample["class"][mask]
          sample["boxes"] = sample["boxes"][mask, :]
          sample["is_crowd"] = sample["is_crowd"][mask]

          yield (sample["id"],
               sample["file_name"],
               sample["class"],
               sample["boxes"])
        else:
          yield (sample["id"],
               sample["file_name"],
               np.array([], dtype=np.int32),
               np.array([[]],dtype=np.float32))
          

  '''
  def parse_fn(self, image_id, file_name, classes, boxes):
    """Parse a single input sample
    """
    image = tf.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.to_float(image)

    scale = [0, 0]
    translation = [0, 0]
    if self.augmenter:
      is_training = (self.config.mode == "train")
      print('is_training = ', is_training)
      image, classes, boxes, scale, translation = self.augmenter(
        image,
        classes,
        boxes,
        self.config.resolution,
        is_training=is_training,
        speed_mode=False, resize_method=self.resize_method)

    return ([image_id], image, classes, boxes, scale, translation, [file_name])

  '''
