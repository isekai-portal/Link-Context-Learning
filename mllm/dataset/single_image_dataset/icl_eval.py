import re
import sys
import logging
import jsonlines
import random
from typing import Dict, Any, Sequence

from .icl_train import ICLTrainDataset, logger

from .. import BaseComputeMetrics
from ..root import (
    DATASETS,
    METRICS
)

@DATASETS.register_module()
class ICLEvalDataset(ICLTrainDataset):
    def __init__(self, policy, sample_per_class = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.sample_per_class = sample_per_class
        self.data_map = self._rearrange()

    def __len__(self):
        return len(self.data_map)

    def _rearrange(self):
        # Map dataloader index to self.data, according to class_idx and sample_idx
        data_map = []
        for cls_idx, item in enumerate(self.data):
            test_samples = item['test_samples']
            for sample_idx, sample in enumerate(test_samples):
                # sample_per_class = 0: all samples evaluation
                if sample_idx == self.sample_per_class and \
                    self.sample_per_class > 0:
                    break
                data_map.append([cls_idx, sample_idx])
        return data_map        

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        class_name = item["class_name"].lower()
        context_samples = item["context_samples"]
        test_samples = item['test_samples']

        test_img = self.get_image(test_samples[sample_idx])
        context_imgs = []
        for i in range(shot):
            context_imgs.append(self.get_image(context_samples[i]))
        return class_name, context_imgs, test_img
    
@DATASETS.register_module()
class LCLEvalDataset(ICLTrainDataset):
    def __init__(self, policy, sample_per_class = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.sample_per_class = sample_per_class    
    
    def __len__(self):
        return len(self.data)
    
    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        class_name = item["class_name"].lower()
        context_samples = item["context_samples"]
        test_samples = item['test_samples']

        test_img = self.get_image(test_samples[sample_idx])
        context_imgs = []
        for i in range(shot):
            context_imgs.append(self.get_image(context_samples[i]))
        return class_name, context_imgs, test_img
    
@METRICS.register_module()
class ICLComputeMetrics(BaseComputeMetrics):
    def extract_ans(self, string: str):
        try:
            found = string.split("ASSISTANT:")[-1].split("</s>")[0].replace("The answer is", "").replace('there is', '').replace('in the image', '').replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None