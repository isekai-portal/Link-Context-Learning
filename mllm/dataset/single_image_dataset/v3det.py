import imp
import sys
import logging
import warnings
import os
import os.path as osp
import random
from typing import Dict, Any, Sequence

import torch
from torchvision.ops import box_iou

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

from pycocotools.coco import COCO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def xywh2xyxy(bbox): 
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def clean_bbox_xyxy(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

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
class V3DetDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.coco = COCO(self.filename)
        self.data = []
        self.cat_ids = self.coco.getCatIds()
        self.cat_ids_set = set(self.cat_ids)
        for icat_id in self.cat_ids:
            cls_info = self.coco.loadCats(icat_id)[0]
            self.data.append(cls_info)


    def __getitem__(self, index):
        if random.random() < 0.5:
            return self.__getitem_pos(index)
        else:
            return self.__getitem_neg(index)

    def __get_icl_item__(self,index,shot):
        ret_list = []
        for i in range(shot):
            ret_list.append(self.__getitem_neg(index))
        for i in range(shot):
            ret_list.append(self.__getitem_pos(index))
        random.shuffle(ret_list)
        ret_list.append(self.__getitem_pos(index))
        return ret_list

    def __getitem_pos(self, index):
        item = self.data[index]
        cls_id = item['id']
        cls_intro = item['cat_info']
        name = item['name']

        img_ids = self.coco.getImgIds(catIds=cls_id)
        img_id = random.choice(img_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cls_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            bbox_xyxy = xywh2xyxy(ann['bbox'])
            bbox_xyxy = clean_bbox_xyxy(bbox_xyxy, width, height)
            bboxes.append(bbox_xyxy)

        expr = random.choice([cls_intro, name])
        img_path = osp.join(self.image_folder, file_name)
        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        ret = {
            'image': image,
            'target': {
                'boxes': bboxes,
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {BOXES_PLACEHOLDER} .',
                    'boxes_seq': [[bbox_idx for bbox_idx in range(len(bboxes))]],
                }
            ]
        }
        return ret
    
    def __getitem_neg(self, index):
        item = self.data[index]
        cls_id = item['id']

        img_ids = self.coco.getImgIds(catIds=cls_id)
        img_id = random.choice(img_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        cat_ids_in_img = set()
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)[0]
            icls_id = ann['category_id']
            cat_ids_in_img.add(icls_id)
        

        # Find neighboring categories not in the image
        neighboring_cat_ids = self.coco['neighbors'][cls_id]
        cur_neighboring_cat_ids = (self.cat_ids_set  - cat_ids_in_img) & neighboring_cat_ids
        cur_neighboring_cat_ids = list(cur_neighboring_cat_ids)

        if len(cur_neighboring_cat_ids) == 0:
            cur_neighboring_cat_ids = random.randint(len(self.cat_ids_set) - 1)

        neighboring_cls_id = random.choice(cur_neighboring_cat_ids)

        neighboring_cls_info = self.coco.loadCats(neighboring_cls_id)[0]
        neighboring_cls_intro = neighboring_cls_info['cat_info']
        neighboring_name = neighboring_cls_info['name']
        neighboring_expr = random.choice([neighboring_cls_intro, neighboring_name])


        img_path = osp.join(self.image_folder, file_name)
        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, neighboring_expr)


        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"The target object does not exist.",
                },
            ]
        }
        return ret
