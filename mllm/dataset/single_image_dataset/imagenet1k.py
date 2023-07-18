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
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class ImageNet1kDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.data = self._get_annos(self.filename)
        self.cls_neg_label = None

    def _get_annos(self, filename):
        cls_metas = []
        with jsonlines.open(filename) as reader:
            for metas in reader:
                cls_metas.append(metas)
        return cls_metas

    def get_raw_item(self, index):
        return self.data[index]

    def get_template(self):
        return 

    def _get_ret(self, index, mode="cls_positive", question=""):
        assert mode in ['cls_positive','cls_negative', 'neighbors']
        assert question != ""

        item = self.get_raw_item(index)
        samples = item['samples']
        neighbors = item['neighbors']
        
        if mode == "cls_positive":
            # current class image and label
            # label = 'shikra'+'_'+item['class_name'].lower()
            label = item['class_name'].lower()
            sample = random.choice(samples)
        elif mode == "cls_negative":
            # current class image, random neighbor label
            if self.cls_neg_label:
                label = self.cls_neg_label
            else:
                metas = random.choice(neighbors)
                # label = 'shikra'
                label = metas[1].lower()
                self.cls_neg_label = label
            sample = random.choice(samples)
        elif mode == "neighbors":
            # random neighbor image and label
            metas = random.choice(neighbors)
            label = metas[1].lower()
            sample = metas[2]
        else:
            raise NotImplementedError

        image = self.get_image(sample)

        # Placeholder for template
        # question = item['text']
        # final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label}.",
                },
            ]
        }
        return ret

    def __getitem__(self, index):
        if random.random() < 0.5:
            ret = self._get_ret(index,mode="cls_positive")
            self.cls_neg_label = None
            return ret 
        else:
            return self._get_ret(index,mode="neighbors")

    # # v4
    # def __get_icl_item__(self, index, shot):
    #     ret_list = []
    #     normal_question = 'What is the class of the image <image>?'
    #     bond_question = 'What is the "bond" class of the image <image>?'
    #     real_question = 'What is the "real" class of the image <image>?'

    #     for _ in range(shot):
    #         ret_list.append(self._get_ret(index, mode = "cls_negative", question=bond_question))

    #     for _ in range(shot):
    #         question = random.choice([bond_question, real_question])
    #         ret_list.append(self._get_ret(index, mode = "neighbors", question=question))
        
    #     random.shuffle(ret_list)
    #     policy = random.choice(["cls_positive", "cls_negative", "neighbors"])
    #     if policy == "cls_positive":
    #         question = real_question
    #     elif policy == "cls_negative":
    #         question = bond_question
    #     elif policy == "neighbors":
    #         question = random.choice([bond_question, real_question])
    #     ret_list.append(self._get_ret(index, mode = policy, question=question))
    #     self.cls_neg_label = None
    #     return ret_list


    # v5
    def __get_icl_item__(self, index, shot):
        ret_list = []
        normal_question = 'What is the class of the image <image>?'
        bond_question = 'What is the "bond" class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            ret_list.append(self._get_ret(index, mode = "cls_negative", question=bond_question))

        for _ in range(shot):
            ret_list.append(self._get_ret(index, mode = "neighbors", question=real_question))
        
        random.shuffle(ret_list)
        policy = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if policy == "cls_positive":
            question = real_question
        elif policy == "cls_negative":
            question = normal_question
        elif policy == "neighbors":
            question = normal_question
        ret_list.append(self._get_ret(index, mode = policy, question=question))
        self.cls_neg_label = None
        return ret_list
