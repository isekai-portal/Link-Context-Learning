from audioop import reverse
from audioop import reverse
from curses.ascii import isdigit
import imp
from pydoc import classname
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

import numpy as np
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

    # v7
    def policy_v7(self, index, shot):
        # Same to v6, only modified the shot to pos_shot and neg_shot
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

        pos_shot = random.randint(1, shot)
        neg_shot = random.randint(0, shot)
        for _ in range(pos_shot):
            image, label = self.get_samples(index, mode = 'cls_positive')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(neg_shot):
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

    # v8
    def policy_v8(self, index, shot):
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
    
    #v9
    def policy_v9(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        mix_question = '[INSTRUCTION] What is in the image <image> ?'
        infer_question = '[INFERENCE] What is in the image <image> ?'

        #shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list


@DATASETS.register_module()
class ImageNet1kDatasetEval(ICLEvalDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        class_name, context_imgs, test_img = self.get_samples(index, shot)
        question, answer = func(class_name, index = index)
        ret_list = []
        # context sample
        for i in range(shot):
            ret_list.append(self.get_ret(context_imgs[i], question=question, answer=answer))
        
        # eval sample
        ret_list.append(self.get_ret(test_img, question=question, answer=answer))
        return ret_list

    def policy_v1(self, class_name, **kargs):
        raise NotImplementedError

    def policy_v2(self, class_name, **kargs):
        raise NotImplementedError
    
    def policy_v3(self, class_name, **kargs):
        question = 'What is the class of the image <image>?'
        answer = class_name
        return question, answer

    def policy_v4(self, class_name, **kargs):
        question = 'What is the "binding" class of the image <image>?'
        answer = class_name
        return question, answer
    
    def policy_v5(self, class_name, **kargs):
        question = 'What is the class of the image <image>?'
        answer = class_name
        return question, answer

    def policy_v6(self, class_name, **kargs):
        question = f'Is there any {class_name} in the image <image> according to the previous conversation ?'
        answer = f'there is {class_name} in the image'
        return question, answer

    def policy_v7(self, class_name, **kargs):
        return self.policy_v6(class_name)

    def policy_v8(self, class_name, **kargs):
        return self.policy_v6(class_name)


@DATASETS.register_module()
class ImageNet1k1WayEval(ICLEvalDataset):
 
    def _rearrange(self):
        wrap_map = []
        data_map = super()._rearrange() #[cls_idx, sample_idx]
        cls_num = len(self.data)
        for i in range(cls_num):
            for data in data_map:
                wrap_map.append([i, data])
        return wrap_map

    def get_samples(self, index, shot):
        ctx_idx, sample_map = self.data_map[index]

        # context samples
        ctx_item = self.get_raw_item(ctx_idx)
        ctx_name = ctx_item["class_name"].lower()
        ctx_samples = ctx_item["context_samples"]
        ctx_imgs = []
        for i in range(shot):
            ctx_imgs.append(self.get_image(ctx_samples[i]))

        # inference sample
        cls_idx, sample_idx = sample_map
        cls_item = self.get_raw_item(cls_idx)
        cls_id = cls_item["class_id"]
        cls_name = cls_item["class_name"].lower()
        cls_samples = cls_item['test_samples']
        infer_img = self.get_image(cls_samples[sample_idx])

        sample_meta = dict(
            ctx_name = ctx_name,
            infer_name = cls_name,
            ctx_imgs = ctx_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)
        QnA = func(sample_meta["ctx_name"], sample_meta["infer_name"])
        
        ret_list = []
        # context sample
        for img in sample_meta["ctx_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["ctx_question"], answer=QnA["ctx_answer"]))

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v9(self, ctx_name, infer_name):
        ctx_question = '[INSTRUCTION] What is in the image <image> ?'
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        ctx_answer = answer.replace(LABEL_PLACEHOLDER, ctx_name)
        infer_answer = answer.replace(LABEL_PLACEHOLDER, infer_name)

        return dict(
            ctx_question = ctx_question, 
            ctx_answer = ctx_answer,
            infer_question = infer_question,
            infer_answer = infer_answer
        )

# context: n-positive samples(class A) + n-negative samples(class B)
# inference: class A or B sample 
@DATASETS.register_module()
class ImageNet1k2WayYNEval(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0
        self.reverse_flag = len(self.data_map)
        self.data_map.extend(self.data_map)

    def rotate_index(self, index):
        # all sample inference twice
        # if reverse = true, context reverse
        reverse = index >= self.reverse_flag
        if reverse:
            index = index - self.reverse_flag
        return index, reverse

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        pos_cls_name = item["class_name"].lower()
        test_samples = item['test_samples']

        # construct positive and negtive pairs
        if cls_idx % 2 == 0:
            neg_cls_idx = cls_idx + 1
        else:
            neg_cls_idx = cls_idx - 1
        neg_item = self.get_raw_item(neg_cls_idx)
        pos_samples = item["context_samples"]
        neg_samples = neg_item["context_samples"]
        neg_cls_name = neg_item["class_name"].lower()
        
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(pos_samples[i]))
            neg_imgs.append(self.get_image(neg_samples[i]))            
        # inference sample (positive class)
        infer_img = self.get_image(test_samples[sample_idx])

        sample_meta = dict(
            pos_cls_name = pos_cls_name,
            neg_cls_name = neg_cls_name,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        index, reverse = self.rotate_index(index)
        sample_meta = self.get_samples(index, shot)

        if reverse:
            QnA = func(sample_meta["neg_cls_name"])
            QnA["infer_answer"] = QnA["infer_answer"].replace(sample_meta["neg_cls_name"], 'no '+ sample_meta["neg_cls_name"])
        else:
            QnA = func(sample_meta["pos_cls_name"])
        
        ret_list = []

        # context sample
        # Note: The traversal process is conducted on a per-class basis.
        # not reverse: pos A image(text: there is A) + neg B image(text: there is no A) + infer A image(label: there is A)
        # reverse: pos A image(text: there is no B) + neg B image(text: there is B) + infer A image(label: there is no B)
        for img in sample_meta["pos_imgs"]:
            if reverse:
                ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
            else:
                ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            if reverse:
                ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
            else:
                ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v6(self, cls_name):
        pos_question = 'What is the charateristic about the image <image> ?'
        neg_question = pos_question
        infer_question = f'Is there any {cls_name} in the image <image> according to the previous conversation ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ cls_name)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )

    def policy_v9(self, cls_name):
        pos_question = '[INSTRUCTION] What is in the image <image> ?'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, 'no'+ cls_name)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )
    
    def policy_v7(self, cls_name):
        return self.policy_v6(cls_name)

    def policy_v8(self, cls_name):
        return self.policy_v6(cls_name)

# context: n-positive samples(class A) + n-negative samples(class B)
# inference: class A or B sample 
@DATASETS.register_module()
class ImageNet1k2WayEval(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        pos_cls_name = item["class_name"].lower()
        test_samples = item['test_samples']

        # construct positive and negtive pairs
        if cls_idx % 2 == 0:
            neg_cls_idx = cls_idx + 1
        else:
            neg_cls_idx = cls_idx - 1
        neg_item = self.get_raw_item(neg_cls_idx)
        pos_samples = item["context_samples"]
        neg_samples = neg_item["context_samples"]
        neg_cls_name = neg_item["class_name"].lower()
        
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(pos_samples[i]))
            neg_imgs.append(self.get_image(neg_samples[i]))            
        # inference sample (positive class)
        infer_img = self.get_image(test_samples[sample_idx])

        sample_meta = dict(
            pos_cls_name = pos_cls_name,
            neg_cls_name = neg_cls_name,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)

        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"])
        
        ret_list = []

        # context sample: pos A image(text: there is A) + neg B image(text: there is B) + infer A image(label: there is A)
        for img in sample_meta["pos_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v9(self, cls_name_pos, cls_name_neg):
        pos_question = '[INSTRUCTION] What is in the image <image> ?'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )
    

@DATASETS.register_module()
class ImageNet1kNWayEval(ICLEvalDataset):
    print("PlaceHolder")


@DATASETS.register_module()
class Test100ZeroShotName(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(policy=None, *args, **kwargs)
        self.question = self.get_question()
    
    def get_question(self):
        q_prefix = f'Here are 100 class: '
        q_body = '' # 'class1, class2, ..., class100.' 
        q_suffix = f'Please answer which class this image <image> belongs to?'

        for idx in range(len(self.data)):
            item = self.data[idx]
            cls_name = item["class_name"].lower()
            q_body += cls_name
            if idx == len(self.data) - 1:
                q_body += '. '
            else:
                q_body += ', '
        
        return q_prefix + q_body + q_suffix 

    def get_answer(self, cls_name):
        return cls_name

    def __get_icl_item__(self, index, shot):
        assert shot == 0
        cls_name, _, infer_img = self.get_samples(index, shot)

        ret_list = []
        # inference
        ret_list.append(self.get_ret(infer_img, question=self.question, answer=cls_name)) 
        return ret_list


@DATASETS.register_module()
class Test100ZeroShotSelect(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(policy=None, *args, **kwargs)
        self.name2id = self.get_name2id()
        self.question = self.get_question()

    def get_name2id(self):
        name2id = dict()
        for idx, item in enumerate(self.data):
            name = item["class_name"].lower()
            name2id[name] = idx
        return name2id
    
    def get_question(self):
        q_prefix = f'Here are 100 class: '
        q_body = '' # '1. class1. 2. class2. ... 100. class100.' 
        q_suffix = f'Please answer which class this image <image> belongs to?'

        for name, idx in self.name2id.items():
            q_body += str(idx) + '. ' + name + '. '
        return q_prefix + q_body + q_suffix 

    def __get_icl_item__(self, index, shot):
        assert shot == 0
        cls_name, _, infer_img = self.get_samples(index, shot)

        ret_list = []
        # inference
        ret_list.append(self.get_ret(infer_img, question=self.question, answer=self.name2id[cls_name])) 
        return ret_list


@METRICS.register_module()
class Test100ZeroShotNameMetrics(ICLComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_names = self.get_classes()

    def get_classes(self):
        cls_names = 'whippet, scottish deerhound, airedale, papillon, cardigan, \
            bighorn, fox squirrel, hartebeest, mongoose, vizsla, komondor, \
            afghan hound, proboscis monkey, white wolf, boston bull, border collie, \
            keeshond, gordon setter, american staffordshire terrier, marmoset, samoyed, \
            madagascar cat, timber wolf, platypus, accordion, starfish, airship, canoe, \
            liner, pirate, motor scooter, steam locomotive, convertible, racer, bassinet, \
            bookcase, medicine chest, park bench, barber chair, studio couch, desk, fig, \
            pineapple, organ, maraca, electric guitar, oboe, valley, sandbar, hammer, brambling, \
            vulture, sulphur-crested cockatoo, toucan, american coot, tench, sturgeon, gar, \
            loggerhead, banded gecko, alligator lizard, green lizard, african crocodile, \
            sea snake, analog clock, printer, pinwheel, barn spider, ground beetle, sea anemone, \
            brain coral, nematode, chiton, dutch oven, rotisserie, mosque, grocery store, barbershop, \
            fountain, bell pepper, head cabbage, spaghetti squash, shower curtain, handkerchief, \
            ashcan, photocopier, crossword puzzle, feather boa, hay, military uniform, dome, barrel, \
            fire screen, pole, overskirt, parachute, sleeping bag, breastplate, stretcher, matchstick'
        cls_names = cls_names.split(', ')
        return cls_names

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:

        name2id = {}
        for id, name in enumerate(self.cls_names):
            name2id[name] = id
            
        correct_num = np.zeros(len(self.cls_names))
        targets_num = np.zeros(len(self.cls_names))
        unknown = 0
        failed = 0
        target_failed = 0
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)

            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue            
            if extract_pred is None:
                failed += 1
            if extract_target not in self.cls_names:
                raise ValueError(f"extract_target error: {extract_target}")
            if extract_pred not in self.cls_names:
                unknown += 1
                continue               
            
            idx = name2id[extract_target]
            targets_num[idx] += 1
            if extract_pred == extract_target:
                correct_num[idx] += 1

        acc = correct_num/targets_num
        acc_str = ''
        for data in acc:
            acc_str += str(data) + ', '
        foot10 = np.argsort(acc)[:10] # 升序
        foot10_name = ''

        for idx in foot10:
            foot10_name += self.cls_names[idx]
            foot10_name += ', '

        return {
            'accuracy': acc,
            'target_failed': target_failed,
            'failed': failed,
            'unknown': unknown,
            'foot10': foot10_name
        }

@METRICS.register_module()
class Test100ZeroShotSelectMetrics(ICLComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_names = self.get_classes()

    def get_classes(self):
        cls_names = 'whippet, scottish deerhound, airedale, papillon, cardigan, \
            bighorn, fox squirrel, hartebeest, mongoose, vizsla, komondor, \
            afghan hound, proboscis monkey, white wolf, boston bull, border collie, \
            keeshond, gordon setter, american staffordshire terrier, marmoset, samoyed, \
            madagascar cat, timber wolf, platypus, accordion, starfish, airship, canoe, \
            liner, pirate, motor scooter, steam locomotive, convertible, racer, bassinet, \
            bookcase, medicine chest, park bench, barber chair, studio couch, desk, fig, \
            pineapple, organ, maraca, electric guitar, oboe, valley, sandbar, hammer, brambling, \
            vulture, sulphur-crested cockatoo, toucan, american coot, tench, sturgeon, gar, \
            loggerhead, banded gecko, alligator lizard, green lizard, african crocodile, \
            sea snake, analog clock, printer, pinwheel, barn spider, ground beetle, sea anemone, \
            brain coral, nematode, chiton, dutch oven, rotisserie, mosque, grocery store, barbershop, \
            fountain, bell pepper, head cabbage, spaghetti squash, shower curtain, handkerchief, \
            ashcan, photocopier, crossword puzzle, feather boa, hay, military uniform, dome, barrel, \
            fire screen, pole, overskirt, parachute, sleeping bag, breastplate, stretcher, matchstick'
        cls_names = cls_names.split(', ')
        return cls_names

    def extract_ans(self, string: str):
        try:
            found = string.split("ASSISTANT:")[-1].split("</s>")[0].replace("The answer is", "").replace("there is", "").replace(".", "").strip()
            return found
        except (IndexError, AttributeError):
            return None

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:

        name2id = {}
        for id, name in enumerate(self.cls_names):
            name2id[name] = id
            
        correct_num = np.zeros(len(self.cls_names))
        targets_num = np.zeros(len(self.cls_names))
        unknown = 0
        failed = 0
        target_failed = 0
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)

            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue            
            if extract_pred is None:
                failed += 1
            cls_ids = [str(i) for i in range(len(self.cls_names))]
            if extract_target not in cls_ids:
                raise ValueError(f"extract_target error: {extract_target}")
            if extract_pred not in cls_ids:
                unknown += 1
                continue
            
            idx = int(extract_target)
            targets_num[idx] += 1
            if extract_pred == extract_target:
                correct_num[idx] += 1

        acc = correct_num/targets_num
        acc_str = ''
        for data in acc:
            acc_str += str(data) + ', '
        foot10 = np.argsort(acc)[:10] # 升序
        foot10_name = ''

        for idx in foot10:
            foot10_name += self.cls_names[idx]
            foot10_name += ', '

        return {
            'accuracy': acc_str,
            'target_failed': target_failed,
            'failed': failed,
            'unknown': unknown,
            'foot10': foot10_name
        }