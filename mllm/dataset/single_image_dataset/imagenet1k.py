import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines
import random
from typing import Dict, Any, Sequence
from .icl_train import ICLTrainDataset, logger
from .icl_eval import ICLEvalDataset
from ..root import DATASETS


@DATASETS.register_module()
class ImageNet1kDatasetTrain(ICLTrainDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        return func(index, shot)

    def policy_v1(self):
        raise NotImplementedError

    def policy_v2(self):
        raise NotImplementedError

    def policy_v3(self, index, shot):
        ret_list = []
        normal_question = 'What is the class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=normal_question, answer=label)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=normal_question, answer=label)
            ret_list.append(ret)

        random.shuffle(ret_list)
        # inference sample
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = normal_question
        elif mode == "neighbors":
            question = normal_question
        
        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    def policy_v4(self, index, shot):
        ret_list = []
        bind_question = 'What is the "binding" class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=bind_question, answer=label)
            ret_list.append(ret)
            
        for _ in range(shot):
            question = random.choice([bind_question, real_question])
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=question, answer=label)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        # inference sample
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = bind_question
        elif mode == "neighbors":
            question = random.choice([bind_question, real_question])

        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    def policy_v5(self, index, shot):
        ret_list = []
        normal_question = 'What is the class of the image <image>?'
        bind_question = 'What is the "binding" class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=bind_question, answer=label)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=real_question, answer=label)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = normal_question
        elif mode == "neighbors":
            question = normal_question

        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    # 2way baseline
    def policy_2way(self, index, shot):
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



@DATASETS.register_module()
class ImageNet1kDatasetEval(ICLEvalDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert policy is not None
        self.policy = policy

    def get_question(self):
        fun = getattr(self, self.policy)
        return fun()

    def policy_v1(self):
        raise NotImplementedError

    def policy_v2(self):
        raise NotImplementedError
    
    def policy_v3(self):
        return 'What is the class of the image <image>?'

    def policy_v4(self):
        return 'What is the "binding" class of the image <image>?'
    
    def policy_v5(self):
        return 'What is the class of the image <image>?'
