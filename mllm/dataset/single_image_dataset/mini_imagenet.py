import imp
import sys
import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from .icl_train import logger
from .icl_eval import ICLComputeMetrics, ICLEvalDataset
from ..utils import MInstrDataset
from .imagenet1k import ImageNet1kDatasetEval

from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

@DATASETS.register_module()
class MiniImageNetDatasetEval(ImageNet1kDatasetEval):
    def _rearrange(self):
        # Map dataloader index to self.data, according to set_idx, cls_idx and sample_idx
        data_map = []
        for set_idx, item in enumerate(self.data):
            test_samples = item['Query']
            for cls_idx, cls_samples in enumerate(test_samples):
                for sample_idx, sample in enumerate(cls_samples):
                    # sample_per_class = 0: all samples evaluation
                    if sample_idx == self.sample_per_class and \
                        self.sample_per_class > 0:
                        break
                    data_map.append([set_idx, cls_idx, sample_idx])
        return data_map     

    def get_samples(self, index, shot):
        set_idx, cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(set_idx)

        support = item['Support'][cls_idx]
        query = item['Query'][cls_idx][sample_idx]
        fake_name = item['Folder'][cls_idx]
        name = item['Name'][cls_idx].lower()
        assert shot <= len(support)

        test_img = self.get_image(query)
        context_imgs = []
        for i in range(shot):
            context_imgs.append(self.get_image(support[i]))
        return name, fake_name, context_imgs, test_img

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        class_name, class_fake_name, context_imgs, test_img = self.get_samples(index, shot)
        question, answer = func(class_name, class_fake_name = class_fake_name, index = index)
        ret_list = []
        # context sample
        for i in range(shot):
            ret_list.append(self.get_ret(context_imgs[i], question=question, answer=answer))
        
        # eval sample
        ret_list.append(self.get_ret(test_img, question=question, answer=answer))
        return ret_list

    def mini_v4(self, class_name, **kargs):
        return super().policy_v4(class_name, **kargs)

    def mini_v5(self, class_name, **kargs):
        return super().policy_v5(class_name, **kargs)

    def mini_v6(self, class_name, class_fake_name, **kargs):
        return super().policy_v6(class_fake_name, **kargs)

    def mini_v7(self, class_name, class_fake_name, **kargs):
        return super().policy_v7(class_fake_name, **kargs)
    
@DATASETS.register_module()
class MiniImageNet5WayEval(ICLEvalDataset):
    raise NotImplementedError