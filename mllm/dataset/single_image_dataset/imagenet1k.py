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
from copy import deepcopy
import numpy as np
import math
import cv2 as cv
from .icl_train import ICLTrainDataset, logger
from .icl_eval import ICLEvalDataset, ICLComputeMetrics, LCLEvalDataset
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
            
            answer = f'there is "{LABEL_PLACEHOLDER}" in the image.'

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
        mix_question = '[INSTRUCTION] What is in the image <image>?'
        infer_question = '[INFERENCE] What is in the image <image>?'

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

    def policy_v9_weight(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image.'

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
        mix_question = '[INSTRUCTION] What is in the image <image>?'
        infer_question = '[INFERENCE] What is in the image <image>?'

        weight = [math.exp((i+1)) for i in range(shot)]
        shot_list = [i+1 for i in range(shot)]
        chosen = random.choices(shot_list,weight)
        shot = chosen[0]
        #shot = random.randint(1, shot)

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
    
    #v9_seq
    #<unk>
    def policy_v9_seq(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode, mask=False):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            
            answer = f'there is "{LABEL_PLACEHOLDER}" in the image'

            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                if not mask:
                    answer = answer.replace(LABEL_PLACEHOLDER, random_string)
                else:
                    answer = answer.replace(LABEL_PLACEHOLDER, " ")
                real_answer = random_string
            elif mode == "neighbors":
                if not mask:
                    answer = answer.replace(LABEL_PLACEHOLDER, label)
                else:
                    answer = answer.replace(LABEL_PLACEHOLDER, " ")
                real_answer = label
            else:
                raise NotImplementedError

            return question, answer, real_answer

        ret_list = []
        mix_question = '[INSTRUCTION] What is in the image <image>?'
        infer_question = '[INFERENCE] Here is the last image <image>. Can you name the categories of all the images that appear above in order?'

        shot = random.randint(1, shot)
        positive_mask = False
        negative_mask = False
        real_ans_list = []
        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            if not positive_mask:
                positive_mask = True
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            else:
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'cls_negative',mask=True)
            real_ans_list.append(real_answer)
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            if not negative_mask:
                negative_mask = True
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            else:
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'neighbors',mask=True)
            real_ans_list.append(real_answer)
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        tmp = list(zip(ret_list,real_ans_list))
        random.shuffle(tmp)
        ret_list,real_ans_list = list(zip(*tmp))
        ret_list = list(ret_list)
        real_ans_list = list(real_ans_list)

        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer, real_answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer, real_answer = _convert_qa(question, label, mode = mode)#use random string
        
        ans_list = [f'"{item}"' for item in real_ans_list]
        final_ans = ', '.join(ans_list)
        final_ans = final_ans + f', "{real_answer}"'
        
        ret = self.get_ret(image, question = question, answer = final_ans)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v9_update
    def policy_v9_update(self, index, shot):
        random_string = None
        random_name_list = []
        def _convert_qa(question, label, mode, empty_mode=False, final=False, wrong_injection=False, order_list=None):
            nonlocal random_string
            nonlocal random_name_list
            assert mode in ['cls_negative', 'neighbors']
            if final:
                if empty_mode:
                    answer = f'{order_list}.'
                else:
                    answer = f'There is "{LABEL_PLACEHOLDER}" in the image.'

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
            else:
                if empty_mode:
                    name = ''.join(random.choices(\
                                string.ascii_uppercase, k=random.randint(1,10))).lower()
                    while name in random_name_list:
                        name = ''.join(random.choices(\
                                    string.ascii_uppercase, k=random.randint(1,10))).lower()
                    random_name_list.append(name)
                    answer = f'The reference name of this image is "{name}".'
                else:
                    answer = f'There is "{LABEL_PLACEHOLDER}" in the image.'

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
        mix_question = '[INSTRUCTION] Tell me something about this image <image>.'
        infer_question = '[INFERENCE] What is in the image <image>?'
        category_question = '[INFERENCE] Which images in the previous example are in the same category as this image <image>? (Provide the answer in order as list)'

        shot = random.randint(3, shot)
        #empty mode
        if random.randint(0,1):
            ori_list = []
            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
                question, answer = _convert_qa(mix_question, label, mode = 'cls_negative',empty_mode=True)
                ret = self.get_ret(image, question = question, answer = answer)
                ret_list.append(ret)
                ori_list.append(0)

            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'neighbors')
                question, answer = _convert_qa(mix_question, label, mode = 'neighbors',empty_mode=True)
                ret = self.get_ret(image, question = question, answer = answer)
                ret_list.append(ret)
                ori_list.append(1)

            tmp = list(zip(ret_list,ori_list,random_name_list))
            random.shuffle(tmp)
            ret_list,ori_list,name_list=list(zip(*tmp))
            ret_list = list(ret_list)
            ori_list = list(ori_list)
            name_list = np.array(list(name_list))

            #ori_seq = np.arange(len(ori_list))
            p_idx = np.array(ori_list)==0
            idx_p = list(name_list[p_idx])

            n_idx = np.array(ori_list)==1
            idx_n = list(name_list[n_idx])
            is_empty = True
        else:
            ori_list = []
            do_inject = random.choice([True,False])
            if do_inject:
                inject_positive = True
                inject_negative = True
            else:
                inject_positive = False
                inject_negative = False
            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
                question, answer = _convert_qa(mix_question, label, mode = 'cls_negative',wrong_injection=inject_positive)
                inject_positive=False
                ret = self.get_ret(image, question = question, answer = answer)
                ret_list.append(ret)
                ori_list.append(0)

            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'neighbors')
                question, answer = _convert_qa(mix_question, label, mode = 'neighbors',wrong_injection=inject_negative)
                inject_negative=False
                ret = self.get_ret(image, question = question, answer = answer)
                ret_list.append(ret)
                ori_list.append(1)

            is_empty = False
        

        if is_empty:
            infer_question = category_question
            mode = random.choice(["cls_negative","neighbors"])
            if mode == "cls_negative":
                question = infer_question
                # need correct label and optional convert to random string
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty ,final=True, order_list=str(idx_p))
            elif mode == "neighbors":
                question = infer_question
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty, final=True, order_list=str(idx_n))
        else:
            mode = random.choice(["cls_negative","neighbors"])
            if mode == "cls_negative":
                question = infer_question
                # need correct label and optional convert to random string
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty ,final=True)
            elif mode == "neighbors":
                question = infer_question
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty, final=True)

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list
    
    #v10
    def policy_v10(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']


            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image'
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)

            elif mode == "neighbors":
                answer = f'there is no "{LABEL_PLACEHOLDER}" in the image'
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        mix_question = '[INSTRUCTION] Learning from the following sentence about the image <image>.'
        infer_question = f'[INFERENCE] Is there any "{LABEL_PLACEHOLDER}" in the image <image>?'

        #shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples(index, mode = "cls_negative")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)#use random string

        question = question.replace(LABEL_PLACEHOLDER, random_string)

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v11
    def policy_v11(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode, infer=False):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']


            if mode == "cls_negative":
                # current class image, random string or current label
                if not infer:
                    answer = f'the image belongs to "type A"'
                else:
                    answer = f'the image belongs to "type A"'

            elif mode == "neighbors":
                if not infer:
                    answer = f'the image belongs to "type B"'
                else:
                    answer = f'the image belongs to "type B"'
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        mix_question = '[INSTRUCTION] Learning from the following sentence about the image <image>.'
        infer_question = f'[INFERENCE] There are two types ("type A" and "type B") of images listed above, which type do you think current image <image> belongs to?'

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

        #question = question.replace(LABEL_PLACEHOLDER, random_string)

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v12
    def policy_v12(self, index, shot):
        Nway = random.randint(2,5)
        if Nway == 2:
            shot = random.randint(1,8)
        elif Nway == 3:
            shot = random.randint(1,5)
        elif Nway == 4:
            shot = random.randint(1,4)
        else:
            shot = random.randint(1,3)
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            
            answer = f'there is "{LABEL_PLACEHOLDER}" in the image'

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
        mix_question = '[INSTRUCTION] What is in the image <image>?'
        infer_question = '[INFERENCE] What is in the image <image>?'

        #shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples_same_cls_Nway(index, mode = 'cls_negative', Nway=Nway)
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image_list, label_list = self.get_samples_same_cls_Nway(index, mode = 'neighbors',Nway=Nway)
            for idx in range(len(image_list)): 
                question, answer = _convert_qa(mix_question, label_list[idx], mode = 'neighbors')#use random string
                ret = self.get_ret(image_list[idx], question = question, answer = answer)
                ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls_Nway(index, mode = "cls_negative", Nway=Nway)
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image_list, label_list = self.get_samples_same_cls_Nway(index, mode = mode, Nway=Nway)
            chosen_i = random.choice(range(len(image_list)))
            image = image_list[chosen_i]
            label = label_list[chosen_i]
            question, answer = _convert_qa(question, label, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v13
    def policy_v13(self, index, shot):
        random_string = None
        random_string_neg = None
        def _convert_qa(question, label, mode, final=False):
            nonlocal random_string
            nonlocal random_string_neg
            assert mode in ['cls_negative', 'neighbors']
            if final:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image.'
            else:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                    random_string_neg = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                    while random_string_neg == random_string:
                        random_string_neg = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(1,10))).lower()
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                    random_string_neg = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                    while random_string_neg == random_string:
                        random_string_neg = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(1,10))).lower()
                #answer = answer.replace(LABEL_PLACEHOLDER, label)
                answer = answer.replace(LABEL_PLACEHOLDER, random_string_neg)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        mix_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        #infer_question = 'What is in the image <image>?'
        infer_question = self.get_template()
        shot = random.randint(1, shot)
        # weight = [math.exp((i+1)/2) for i in range(shot)]
        # shot_list = [i+1 for i in range(shot)]
        # chosen = random.choices(shot_list,weight)
        # shot = chosen[0]
        
        if random.randint(0,1):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot - 1
            shot_n = shot
        else:
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot
            shot_n = shot - 1

        tmp_list = []
        for _ in range(shot_p):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)

        for _ in range(shot_n):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)
        
        random.shuffle(tmp_list)
        ret_list = ret_list + tmp_list

        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer = _convert_qa(question, label, mode = "cls_negative", final=True)
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode, final=True)#use random string

        ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v13_seq
    def policy_v13_seq(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode, final=False, mask=False):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            if final:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image.'
            else:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                if not mask:
                    answer = answer.replace(LABEL_PLACEHOLDER, random_string)
                else:
                    answer = answer.replace(LABEL_PLACEHOLDER, " ")
                real_answer = random_string
            elif mode == "neighbors":
                if not mask:
                    answer = answer.replace(LABEL_PLACEHOLDER, label)
                else:
                    answer = answer.replace(LABEL_PLACEHOLDER, " ")
                real_answer = label
            else:
                raise NotImplementedError

            return question, answer, real_answer

        ret_list = []
        mix_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        #infer_question = 'What is in the image <image>?'
        infer_question = 'Here is the last image <image>. Can you name the categories of all the images that appear above in order?'
        shot = random.randint(1, shot)
        
        
        if random.randint(0,1):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            question, answer, real_answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot - 1
            shot_n = shot
        else:
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer, real_answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot
            shot_n = shot - 1

        positive_mask = False
        negative_mask = False
        real_ans_list = []
        tmp_list = []
        for _ in range(shot_p):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            if not positive_mask:
                positive_mask = True
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            else:
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'cls_negative', mask=True)
            real_ans_list.append(real_answer)
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)

        for _ in range(shot_n):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            if not negative_mask:
                negative_mask = True
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            else:
                question, answer, real_answer = _convert_qa(mix_question, label, mode = 'neighbors',mask=True)#use random string
            real_ans_list.append(real_answer)
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)

        tmp = list(zip(tmp_list,real_ans_list))
        random.shuffle(tmp)
        tmp_list,real_ans_list = list(zip(*tmp))
        tmp_list = list(tmp_list)
        real_ans_list = list(real_ans_list)
        
        ret_list = ret_list + tmp_list

        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer, real_answer = _convert_qa(question, label, mode = "cls_negative", final=True)
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer, real_answer = _convert_qa(question, label, mode = mode, final=True)#use random string

        ans_list = [f'"{item}"' for item in real_ans_list]
        final_ans = ', '.join(ans_list)
        final_ans = final_ans + f', "{real_answer}"'

        ret = self.get_ret_raw(image, question = question, answer = final_ans, conv_mode="final_v1.0")
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v9_update
    def policy_v13_update(self, index, shot):
        random_string = None
        random_name_list = []
        def _convert_qa(question, label, mode, empty_mode=False, final=False, wrong_injection=False, order_list=None):
            nonlocal random_string
            nonlocal random_name_list
            assert mode in ['cls_negative', 'neighbors']
            if final:
                if empty_mode:
                    answer = f'{order_list}.'+'_'+random_string
                else:
                    answer = f'There is "{LABEL_PLACEHOLDER}" in the image.'

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
            else:
                if empty_mode:
                    if random_string:
                        #label = random_string
                        pass
                    else:
                        random_string = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(1,10))).lower()
                    name = ''.join(random.choices(\
                                string.ascii_uppercase, k=random.randint(1,10))).lower()
                    while name in random_name_list:
                        name = ''.join(random.choices(\
                                    string.ascii_uppercase, k=random.randint(1,10))).lower()
                    name = name+'_'+random_string
                    random_name_list.append(name)
                    answer = f'The reference name of this image is "{name}". [END EXAMPLE]'
                else:
                    answer = f'There is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'

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
        mix_question = '[BEGIN EXAMPLE] Tell me something about this image <image>.'
        infer_question = self.get_template()
        category_question = 'Which images in the previous example are in the same category as this image <image>? (Provide the answer in order as list)'

        shot = random.randint(3, shot)
        #empty mode
        if random.randint(0,1):
            ori_list = []
            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
                question, answer = _convert_qa(mix_question, label, mode = 'cls_negative',empty_mode=True)
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
                ret_list.append(ret)
                ori_list.append(0)

            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'neighbors')
                question, answer = _convert_qa(mix_question, label, mode = 'neighbors',empty_mode=True)
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
                ret_list.append(ret)
                ori_list.append(1)

            tmp = list(zip(ret_list,ori_list,random_name_list))
            random.shuffle(tmp)
            ret_list,ori_list,name_list=list(zip(*tmp))
            ret_list = list(ret_list)
            ori_list = list(ori_list)
            name_list = np.array(list(name_list))

            #ori_seq = np.arange(len(ori_list))
            p_idx = np.array(ori_list)==0
            idx_p = list(name_list[p_idx])

            n_idx = np.array(ori_list)==1
            idx_n = list(name_list[n_idx])
            is_empty = True
        else:
            ori_list = []
            do_inject = random.choice([True,False])
            if do_inject:
                inject_positive = True
                inject_negative = True
            else:
                inject_positive = False
                inject_negative = False
            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
                question, answer = _convert_qa(mix_question, label, mode = 'cls_negative',wrong_injection=inject_positive)
                inject_positive=False
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
                ret_list.append(ret)
                ori_list.append(0)

            for _ in range(shot):
                image, label = self.get_samples_same_cls(index, mode = 'neighbors')
                question, answer = _convert_qa(mix_question, label, mode = 'neighbors',wrong_injection=inject_negative)
                inject_negative=False
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
                ret_list.append(ret)
                ori_list.append(1)

            is_empty = False
        ret_list[0]['mode'] = 'causal_v1.0'

        if is_empty:
            infer_question = category_question
            mode = random.choice(["cls_negative","neighbors"])
            if mode == "cls_negative":
                question = infer_question
                # need correct label and optional convert to random string
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty ,final=True, order_list=str(idx_p))
            elif mode == "neighbors":
                question = infer_question
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty, final=True, order_list=str(idx_n))
        else:
            mode = random.choice(["cls_negative","neighbors"])
            if mode == "cls_negative":
                question = infer_question
                # need correct label and optional convert to random string
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty ,final=True)
            elif mode == "neighbors":
                question = infer_question
                image, label = self.get_samples_same_cls(index, mode = mode)
                question, answer = _convert_qa(question, label, mode = mode, empty_mode=is_empty, final=True)

        ret = self.get_ret(image, question = question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #v14
    def policy_v14(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode, final=False):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            if final:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image.'
            else:
                answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
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
        mix_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        #infer_question = 'What is in the image <image>?'
        infer_question = self.get_template()
        #shot = random.randint(1, shot)
        weight = [math.exp((i+1)/2) for i in range(shot)]
        shot_list = [i+1 for i in range(shot)]
        chosen = random.choices(shot_list,weight)
        shot = chosen[0]
        
        if random.randint(0,1):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot - 1
            shot_n = shot
        else:
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot
            shot_n = shot - 1

        tmp_list = []
        for _ in range(shot_p):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)

        for _ in range(shot_n):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)
        
        random.shuffle(tmp_list)
        ret_list = ret_list + tmp_list

        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer = _convert_qa(question, label, mode = "cls_negative", final=True)
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode, final=True)#use random string

        ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list

    #jigsaw_v1
    def policy_jigsaw_v1(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode, final=False):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            if final:
                answer = f'There is "{LABEL_PLACEHOLDER}" in the image.'
            else:
                answer = f'There is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
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
        mix_question = '[BEGIN EXAMPLE] This image <image> is a puzzle piece.'
        infer_question = 'Using the puzzle pieces provided above to piece together this image <image>, and provide the coordinates of each piece on the image coordinate system in order.'
        #infer_question = self.get_template()

        tiles_list = [4,6,9,12,16]
        combination_dict = {4:[[2,2]],6:[[2,3],[3,2]],9:[[3,3]],12:[[3,4],[4,3]],16:[[4,4]]}
        N = random.choice(tiles_list)
        comb = combination_dict[N]
        comb = random.choice(comb)
        #shot = random.randint(1, shot)
        image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
        #whole_image = deepcopy(image)

        tiles = [None]*N
        width, height = image.size
        canvas = np.zeros((height,width,3))
        split_w,split_h = comb
        w_element = int(width/split_w)
        h_element = int(height/split_h)
        order_list = []

        meta_order_list = []
        cnt = 0
        for i in range(split_w):
            for j in range(split_h):
                tmp = image.crop([i*w_element,j*h_element,(i+1)*w_element,(j+1)*h_element])
                tiles[cnt] = tmp
                cnt += 1
                order_list.append([i,j])
                meta_order_list.append([i,j])

        tmp = list(zip(tiles,order_list))
        random.shuffle(tmp)
        tile,order_list=list(zip(*tmp))

        sub_cnt = 0
        for j in range(split_h):
            for i in range(split_w):
                cv_img = cv.cvtColor(np.array(tile[sub_cnt]),cv.COLOR_RGB2BGR)
                canvas[j*h_element:(j+1)*h_element,i*w_element:(i+1)*w_element] = cv_img
                sub_cnt += 1

        tmp = list(zip(tile,meta_order_list))
        random.shuffle(tmp)
        tile,meta_order_list=list(zip(*tmp))
        

        for i in range(len(tile)):
            answer = '[END EXAMPLE]'
            question = mix_question
            image = tile[i]
            if i == 0:
                ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="causal_v1.0")    
            else:
                ret = self.get_ret_raw(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            ret_list.append(ret)
        
        answer = 'The order is: '+str(list(meta_order_list))+'.'
        ret = self.get_ret_raw(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
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

        answer = f'there is {LABEL_PLACEHOLDER} in the image.'
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

        # for c_idx in range(1):
        # c_idx = 14
        # ans = f'there is {LABEL_PLACEHOLDER} in the image'
        # ans = ans.replace(LABEL_PLACEHOLDER,''.join(random.choices(\
        #                     string.ascii_uppercase, k=12)).lower())
        # ret_list[c_idx]['conversations'][1]['value'] = ans
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

    def policy_v9_update(self, cls_name_pos, cls_name_neg):
        pos_question = '[INSTRUCTION] Tell me something about this image <image>.'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is "{LABEL_PLACEHOLDER}" in the image'
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
class ImageNetISEKAI2wayEval(LCLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0

    def get_samples_isekai(self, index, shot):
        item = self.get_raw_item(index)
        img_name = item['image_name']
        positive_list = item['positive_list']
        negative_list = item['negative_list']
        positive_prompt = item['positive_promt']
        negative_prompt = item['negative_promt']
        label_name = item['label_name']
        
        infer_img = self.get_image(img_name)
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(positive_list[i]))
            neg_imgs.append(self.get_image(negative_list[i])) 

        sample_meta = dict(
            pos_cls_name = positive_prompt,
            neg_cls_name = negative_prompt,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img,
            label_name = label_name,
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples_isekai(index, shot)

        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"],sample_meta["label_name"])
        
        ret_list = []

        # context sample: pos A image(text: there is A) + neg B image(text: there is B) + infer A image(label: there is A)
        for img in sample_meta["pos_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        # for c_idx in range(1):
        # c_idx = 14
        # ans = f'there is {LABEL_PLACEHOLDER} in the image'
        # ans = ans.replace(LABEL_PLACEHOLDER,''.join(random.choices(\
        #                     string.ascii_uppercase, k=12)).lower())
        # ret_list[c_idx]['conversations'][1]['value'] = ans
        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v9(self, cls_name_pos, cls_name_neg, label_name):
        pos_question = '[INSTRUCTION] What is in the image <image> ?'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = answer.replace(LABEL_PLACEHOLDER,label_name)

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )

@DATASETS.register_module()
class ImageNet1k2WayCleanISEKAIEval(LCLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0

    def get_samples(self, index, shot):
        item = self.get_raw_item(index)
        img_name = item['image_name']
        positive_list = item['positive_list']
        negative_list = item['negative_list']
        positive_prompt = item['positive_promt']
        negative_prompt = item['negative_promt']
        label_name = item['label_name']
        
        infer_img = self.get_image(img_name)
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(positive_list[i]))
            neg_imgs.append(self.get_image(negative_list[i])) 

        sample_meta = dict(
            pos_cls_name = positive_prompt,
            neg_cls_name = negative_prompt,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img,
            label_name = label_name,
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)

        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"],sample_meta["label_name"])
        
        ret_list = []

        # context sample: pos A image(text: there is A) + neg B image(text: there is B) + infer A image(label: there is A)
        if shot > 0:
            if random.randint(0,1):
                img = sample_meta["pos_imgs"].pop(0)
                ret_list.append(self.get_ret_raw(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="causal_v1.0"))
                
            else:
                img = sample_meta["neg_imgs"].pop(0)
                ret_list.append(self.get_ret_raw(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="causal_v1.0"))
                

            tmp_list = []
            for img in sample_meta["pos_imgs"]:
                tmp_list.append(self.get_ret_raw(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="hypnotized_ans_v1.0"))
            
            for img in sample_meta["neg_imgs"]:
                tmp_list.append(self.get_ret_raw(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="hypnotized_ans_v1.0"))
            random.shuffle(tmp_list)
            ret_list = ret_list + tmp_list

        ret_list.append(self.get_ret_raw(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
        return ret_list

    def policy_v13(self, cls_name_pos, cls_name_neg, label_name):
        pos_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        neg_question = pos_question
        infer_question = f'Based on the previous examples, what is in the image <image>?'

        #answer = f'there is {LABEL_PLACEHOLDER} in the image'
        answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = answer.replace(LABEL_PLACEHOLDER,label_name)

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )
    
    

@DATASETS.register_module()
class ImageNet1k2WayCleanEval(ICLEvalDataset):
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
        if shot > 0:
            if random.randint(0,1):
                img = sample_meta["pos_imgs"].pop(0)
                ret_list.append(self.get_ret_raw(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="causal_v1.0"))
                
            else:
                img = sample_meta["neg_imgs"].pop(0)
                ret_list.append(self.get_ret_raw(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="causal_v1.0"))
                

            tmp_list = []
            for img in sample_meta["pos_imgs"]:
                tmp_list.append(self.get_ret_raw(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="hypnotized_ans_v1.0"))
            
            for img in sample_meta["neg_imgs"]:
                tmp_list.append(self.get_ret_raw(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="hypnotized_ans_v1.0"))
            random.shuffle(tmp_list)
            ret_list = ret_list + tmp_list

        # c_idx = 2
        # ans = f'there is {LABEL_PLACEHOLDER} in the image'
        # ans = ans.replace(LABEL_PLACEHOLDER,''.join(random.choices(\
        #                     string.ascii_uppercase, k=random.randint(1,10))).lower())
        # ret_list[c_idx]['conversations'][1]['value'] = ans
        # for c_idx in range(16):
        #     ans = f'there is {LABEL_PLACEHOLDER} in the image'
        #     ans = ans.replace(LABEL_PLACEHOLDER,''.join(random.choices(\
        #                         string.ascii_uppercase, k=random.randint(1,10))).lower())
        #     ret_list[c_idx]['conversations'][1]['value'] = ans
        # inference
        ret_list.append(self.get_ret_raw(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
        return ret_list

    def policy_v13(self, cls_name_pos, cls_name_neg):
        pos_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        neg_question = pos_question
        infer_question = f'Based on the previous examples, what is in the image <image>?'

        #answer = f'there is {LABEL_PLACEHOLDER} in the image'
        answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
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
    
@METRICS.register_module()
class ImageNet2WayMetrics(ICLComputeMetrics):
    def extract_ans(self, string: str):
        try:
            found = string.split("ASSISTANT:")[-1].split("<PAD>")[0].replace("The answer is", "")
            found = found.split('there is')[-1].replace('in the image', '').replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        # pd = data['pred'].split(' "target"')[0].lower()
        # gt = data['target'].split('ASSISTANT:')[-1].split('</s>')[0].lower()
        # gt_label = gt.split(' there is ')[1].split(' ')[0]
        # if gt_label in pd:
        #     correct+=1 

        correct = 0
        failed = 0
        target_failed = 0
        for pred, target in zip(preds, targets):
            # extract_pred = self.extract_ans(pred)
            extract_pred = pred
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1

            if extract_target in extract_pred:
                correct += 1
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }



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
        ret_list.append(self.get_ret(infer_img, question=self.question, answer=self.name2id[cls_name], conv_mode='icl_v2.0')) 
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