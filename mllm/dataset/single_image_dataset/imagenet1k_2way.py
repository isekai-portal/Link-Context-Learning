import imp
from re import template
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
from .imagenet import ImageNetDataset

from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class ImageNet1k2WayDataset(ImageNetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ret(self, image, question, answer):
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"{answer}",
                },
            ]
        }
        return ret

    def __get_icl_item__(self, index, shot):
        cls_label = self.get_raw_item(index)['class_name'].lower()
        question = f"Is there any {cls_label} in this image <image>?"

        ret_list = []
        # positive sample
        for i in range(shot):
            image, label = self.get_samples(index, mode = "cls_positive")
            ret_list.append(self.get_ret(image, question, answer="Yes"))
        
        # negative sample
        for i in range(shot):
             image, label = self.get_samples(index, mode = "neighbors")
             ret_list.append(self.get_ret(image, question, answer="No"))

        random.shuffle(ret_list)
        policy = ['cls_positive', 'neighbors']
        mode = random.choice(policy)
        if mode == 'cls_positive':
            ret_list.append(self.get_ret(image, question, answer="Yes"))
        else:
            ret_list.append(self.get_ret(image, question, answer="No"))
        
        return ret_list
