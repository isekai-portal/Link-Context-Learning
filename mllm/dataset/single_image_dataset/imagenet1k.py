import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from typing import Dict, Any, Sequence
from .icl_train import ICLTrainDataset, logger
from .icl_eval import ICLEvalDataset, ICLComputeMetrics
from ..root import (
    DATASETS,
    METRICS,
    EXPR_PLACEHOLDER
)
LABEL_PLACEHOLDER = "<label>"

@DATASETS.register_module()
class ImageNet1kDatasetTrain(ICLTrainDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
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

    # v6
    def policy_v6(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_positive','cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_positive":
                # current class image, current label
                question = question.replace(LABEL_PLACEHOLDER, label)
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            elif mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    label = random_string
                else:
                    if random.random() < 0.5:
                        label = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(4,8))).lower()
                    random_string = label
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                # random neighbor image and label
                assert random_string is not None
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ random_string)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        correct_question = 'Forget about the previous conversation, Is there any <label> in the image <image> ?'
        mix_question = 'What is the charateristic about the image <image> ?'
        infer_question = 'Is there any <label> in the image <image> according to the previous conversation ?'

        shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'cls_positive')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label = None, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive","cls_negative","neighbors"])
        if mode == "cls_positive":
            question = correct_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)
        elif mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples(index, mode = "cls_positive")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label = None, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
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
        self.policy = policy

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        question = func(index, shot)

        class_name, context_imgs, test_img = self.get_samples(index, shot)
        ret_list = []
        # context sample
        for i in range(shot):
            ret_list.append(self.get_ret(context_imgs[i], question=question, answer=class_name))
        
        # eval sample
        ret_list.append(self.get_ret(test_img, question=question, answer=class_name))
        return ret_list

    def policy_v1(self, *args):
        raise NotImplementedError

    def policy_v2(self, *args):
        raise NotImplementedError
    
    def policy_v3(self, *args):
        return 'What is the class of the image <image>?'

    def policy_v4(self, *args):
        return 'What is the "binding" class of the image <image>?'
    
    def policy_v5(self, *args):
        return 'What is the class of the image <image>?'

    # 2way baseline
    def policy_2way(self, index, *args):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        cls_label = item['class_name'].lower()
        question = f"Is there any {cls_label} in this image <image>?"
        return question

@METRICS.register_module()
class ImageNet1kComputeMetrics(ICLComputeMetrics):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def extract_ans(self, string: str, mode: str):
        ans = super().extract_ans(string, mode)
        func = getattr(self, self.policy)
        if(isfunction(func)):
            ans = func(ans)
        else:
            logger.warning(f"Policy {self.policy} not exists, calling ICLComputeMetrics process.")
        return ans

    # 2way baseline
    def policy_2way(self, ans):
        return question