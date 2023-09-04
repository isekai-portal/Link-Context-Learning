import imp
import sys
import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from .lcl import LCLDataset, LCLComputeMetrics, logger, LABEL_PLACEHOLDER
from ..root import (
    DATASETS,
    METRICS,)


@DATASETS.register_module()
class ISEKAIEval(LCLDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
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
        raise NotImplementedError


@DATASETS.register_module()
class ISEKAIEval2Way(ISEKAIEval):

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)
        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"],sample_meta["label_name"])
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

        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
        return ret_list

    def policy_2way(self, cls_name_pos, cls_name_neg, label_name):
        pos_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        neg_question = pos_question
        infer_question = f'Based on the previous examples, what is in the image <image>?'

        answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = answer.replace(LABEL_PLACEHOLDER, label_name).replace(' [END EXAMPLE]', '')

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )

@METRICS.register_module()
class ISEKAIMetrics(LCLComputeMetrics):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(filename, *args, **kwargs)
        self.gt_pairs = self.get_pairs_isekai()

    def get_pairs_isekai(self):
        target_pairs = []
        with jsonlines.open(self.filename) as reader:
            for metas in reader:
                positive_prompt = metas['positive_promt'].lower()
                negative_prompt = metas['negative_promt'].lower()
                target_pairs.append([positive_prompt, negative_prompt])
        return target_pairs    

    def get_neg_pair(self, index, target):
        pair = self.gt_pairs[index]

        pos_target = target
        for name in pair:
            if name != pos_target:
                neg_target = name
        return neg_target