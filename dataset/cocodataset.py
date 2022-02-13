import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *
from utils.aug_utils import *

coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )

class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self,cfg,mode='train',min_size=1,debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            cfg (dict) : Configuration file.
                model_type (str): model name specified in config file
                data_dir (str): dataset root directory
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.cfg = cfg 
        self.data_dir = self.cfg['DATA']['DATADIR']
        self.model_type = self.cfg['MODEL']['TYPE']

        if mode == 'train':
            self.json_file = 'instances_train2017.json'
            self.name = 'train2017'
            self.img_size = self.cfg['TRAIN']['IMGSIZE']
            self.augmentation = self.cfg['AUGMENTATION']

        elif mode == 'val':
            self.json_file = 'instances_val2017.json'
            self.name = 'val2017'
            self.img_size = self.cfg['TEST']['IMGSIZE']
            self.augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}
        
        self.coco = COCO(os.path.join(self.data_dir,'annotations/'+self.json_file))
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())

        self.max_labels = 50
        self.min_size = min_size

        self.img_file = os.path.join(self.data_dir, self.name,'{:012}' + '.jpg')

        self.lrflip = False
        if self.augmentation['LRFLIP'] and np.random.rand() > 0.5 == True:
            self.lrflip = True
    @property
    def coco_class_ids(self):
        return coco_class_ids
    @property
    def coco_label_names(self):
        return coco_label_names

    def __len__(self):
        return len(self.ids)
    
    def preprocess(self,img):
        img, info_img = preprocess(img, self.img_size, jitter=self.augmentation['JITTER'],
                                   random_placing=self.augmentation['RANDOM_PLACING'])
        if self.augmentation['RANDOM_DISTORT']:
            img = random_distort(img,self.augmentation['HUE'],self.augmentation['SATURATION'],self.augmentation['EXPOSURE'])
        img = np.transpose(img / 255., (2, 0, 1))
        if self.lrflip:
            img = np.flip(img, axis=2).copy()
        return img, info_img
    
    def get_labels(self,annotations,info_img):
        labels = []
        padded_labels = np.zeros((self.max_labels, 5))
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels,info_img, self.img_size,self.lrflip)
            padded_labels[range(len(labels))[:self.max_labels]] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)
        return padded_labels 

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img = cv2.imread(self.img_file.format(id_))

        # if self.json_file == 'instances_val5k.json' and img is None:
        #     img_file = os.path.join(self.data_dir, 'train2017',
        #                             '{:012}'.format(id_) + '.jpg')
        #     img = cv2.imread(img_file)

        assert img is not None
        img, info_img = self.preprocess(img)
        
        # load labels
        padded_labels = self.get_labels(annotations,info_img)
        
        return img, padded_labels, info_img, id_
    
    def random_resize(self):
        imgsize = (random.randint(0, 9) % 10 + 10) * 32
        self.img_size = imgsize
        print("New img_size set to ",imgsize)
