import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines
import random
from typing import Dict, Any, Sequence
from numpy import real

import torch
from torchvision.ops import box_iou

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    BOXES_PLACEHOLDER,
)
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def image_crop(img,bbox):
    crop_img = img.crop((int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
    return crop_img

def clean_bbox_xyxy(bbox, img_w, img_h):
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    if x1 > x2:
        x2 = x1 + 1
    if y1 > y2:
        y2 = y1 + 1

    if x1 < 0:
        x1 = 0.0
    if x2 > img_w:
        x2 = float(img_w)
    if y1 < 0:
        y1 = 0.0
    if y2 > img_h:
        y2 = float(img_h)
        
    return [x1, y1, x2, y2]

@DATASETS.register_module()
class BusinessDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.data = self._get_annos(self.filename)
        self.data_positive = self._get_annos(self.filename_positive)
        self.data_negative = self._get_annos(self.filename_negative)

        self.cls_neg_label = None

    def _get_annos(self, filename):
        cls_metas = []
        with jsonlines.open(filename) as reader:
            for metas in reader:
                cls_metas.append(metas)
        return cls_metas

    def get_raw_item(self, index, is_icl,mode=None):
        if not is_icl:
            return self.data[index]
        else:
            if mode == 'cls_positive':
                return self.data_positive[index]
            else:
                return self.data_negative[index]


    def get_template(self):
        return 

    def get_image(self, image_folder, image_path):
        if image_folder is not None:
            image_path = os.path.join(image_folder, image_path)
        image = Image.open(image_path).convert('RGB')
        return image

    def _get_ret_origin(self, index, mode):


        item = self.get_raw_item(index,is_icl=False)
        try:
            img_id = item['filename']
        except:
            img_id = item['image_id']
        

        Q3 = 'Is there any <exp> in the image <image> ?'
        final_question = Q3
        if self.label_name is not None:
            label = self.label_name

        final_question = final_question.replace('<exp>',label)

        if mode == 'cls_positive':
            image = self.get_image(self.image_folder_positive,img_id)
        else:
            image = self.get_image(self.image_folder_negative,img_id)
        try:
            box = item['boxes'][0]
            if len(box) != 0:
                image = image_crop(image,box)
        except:
            pass

        if mode == 'cls_positive':
            label = 'yes'
        else:
            label = 'no'

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label}.",
                },
            ]
        }
        return ret
    
    def __getitem__(self, index):
        if self.data[index] in self.data_positive:
            return self._get_ret_origin(index, mode = "cls_positive")
        else:
            return self._get_ret_origin(index, mode = "cls_negative")

    def __get_icl_item__(self, index, shot):
        ret_list = []

        if shot != 0:
            ret_list.append(self._get_ret(0, mode = "cls_positive", conv_mode='hypnotized_v1.0'))

            tmp_list = []
            tmp_idx = 1
            for _ in range(shot-1):
                tmp_list.append(self._get_ret(tmp_idx, mode = "cls_positive", conv_mode='hypnotized_ans_v1.0'))
                tmp_idx += 1

            tmp_idx = 0
            for _ in range(shot):
                tmp_list.append(self._get_ret(tmp_idx, mode = "cls_negative", conv_mode='hypnotized_ans_v1.0'))
                tmp_idx += 1
            
            random.shuffle(tmp_list)
            ret_list = ret_list + tmp_list

            if self.data[index] in self.data_positive:
                ret_list.append(self._get_ret(index, mode = "cls_positive",is_icl=False, conv_mode='hypnotized_v1.1'))
            else:
                ret_list.append(self._get_ret(index, mode = "cls_negative",is_icl=False, conv_mode='hypnotized_v1.1'))
        else:
            if self.data[index] in self.data_positive:
                ret_list.append(self._get_ret(index, mode = "cls_positive",is_icl=False, conv_mode='vicuna_v1.1'))
            else:
                ret_list.append(self._get_ret(index, mode = "cls_negative",is_icl=False, conv_mode='vicuna_v1.1'))

        return ret_list


    def _get_ret(self, index, mode="cls_positive", is_icl=True, conv_mode=None):
        assert mode in ['cls_positive','cls_negative']

        item = self.get_raw_item(index,is_icl,mode)
        try:
            img_id = item['filename']
        except:
            img_id = item['image_id']
        
        normal_question ='[INSTRUCTION] Please learn from the following sentence about the image <image>.'
        bind_question = "[INFERENCE] If '<exp>' is in this picture <image>, could you identify its location?"

        try:
            box = item['bbox']
        except:
            try:
                box = item['instances'][0]['bbox']
            except:
                pass


        #Q3 = 'What is the charateristic about the image <image> ?'
        #label = 'Credential_file'
        if self.label_name is not None:
            label = self.label_name
        Q3 = normal_question
        Q3 = Q3.replace('<exp>',label)
        #bind_question = 'Is there any <exp> in the image <image> ?'
        bind_question = bind_question.replace('<exp>',label)
        if not is_icl:
            final_question = bind_question
            #image = self.get_image(self.image_folder,img_id)
            if mode == 'cls_positive':
                image = self.get_image(self.image_folder_positive,img_id)
                label = f'{BOXES_PLACEHOLDER}'
            else:
                image = self.get_image(self.image_folder_negative,img_id)
                label = 'The target object does not exist.'
        else:
            if mode == 'cls_positive':
                final_question = Q3
                image = self.get_image(self.image_folder_positive,img_id)
                label = f'The {BOXES_PLACEHOLDER} in the image is ' + label +'.'
            else:
                final_question = Q3
                image = self.get_image(self.image_folder_negative,img_id)
                label = 'The target object does not exist.'

        w, h = image.size
        bboxes = []

        bbox_xyxy = clean_bbox_xyxy(box, w, h)
        bboxes.append(bbox_xyxy)

        
        # if mode == 'cls_positive':
        #     label = 'ID card'
        # else:
        #     label = 'not ID card'
        if mode == 'cls_positive':
            if not is_icl:
                ret = {
                    'image': image,
                    'target': {
                        'boxes': bboxes,
                    },
                    'conversations': [
                        {
                            'from': 'human',
                            'value': final_question,
                        },
                        {
                            'from': 'gpt',
                            'value': label,
                            'boxes_seq': [[bbox_idx for bbox_idx in range(len(bboxes))]],
                        },],
                    'mode': conv_mode
                }
            else:
                ret = {
                    'image': image,
                    'target': {
                        'boxes': bboxes,
                    },
                    'conversations': [
                        {
                            'from': 'human',
                            'value': final_question,
                        },
                        {
                            'from': 'gpt',
                            'value': label,
                            'boxes_seq': [[bbox_idx for bbox_idx in range(len(bboxes))]],
                        },],
                    'mode': conv_mode
                }
        else:
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                    },
                    {
                        'from': 'gpt',
                        'value': label,
                    },],
                'mode': conv_mode
            }
        return ret
