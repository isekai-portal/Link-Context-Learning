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
from copy import deepcopy
import numpy as np
import math
import cv2 as cv
from .lcl import LCLDataset, logger, LABEL_PLACEHOLDER
from ..root import (
    DATASETS,
    METRICS,
    EXPR_PLACEHOLDER
)

@DATASETS.register_module()
class ImageNet1kDatasetTrain(LCLDataset):   
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        return func(index, shot)

    #v13
    def policy_v13(self, index, shot):
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
        
        if random.randint(0, 1):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot - 1
            shot_n = shot
        else:
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer, conv_mode="causal_v1.0")
            ret_list.append(ret)
            shot_p = shot
            shot_n = shot - 1

        tmp_list = []
        for _ in range(shot_p):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            tmp_list.append(ret)

        for _ in range(shot_n):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
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

        ret = self.get_ret(image, question = question, answer = answer, conv_mode="final_v1.0")
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

    #jigsaw_v1
    def policy_jigsaw_v1(self, index, shot):
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
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="causal_v1.0")    
            else:
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            ret_list.append(ret)
        
        answer = 'The order is: '+str(list(meta_order_list))+'.'
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list


@DATASETS.register_module()
class ImageNetTest100Eval(LCLDataset):
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
class ImageNetTest100Eval2Way(ImageNetTest100Eval):
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

        for i in range(len(ret_list)):
            if i == 0:
                conv_mode = 'causal_v1.0'
            else:
                conv_mode = 'hypnotized_ans_v1.0'
            ret_list[i]['mode'] = conv_mode

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
        return ret_list

    def policy_v13(self, cls_name_pos, cls_name_neg):
        pos_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        neg_question = pos_question
        infer_question = f'Based on the previous examples, what is in the image <image>?'

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
