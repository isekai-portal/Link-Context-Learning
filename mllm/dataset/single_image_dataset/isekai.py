import imp
import sys
import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from .icl_train import logger
from .icl_eval import LCLEvalDataset
from .imagenet1k import ImageNet1kDatasetEval

from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
LABEL_PLACEHOLDER = "<label>"

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
                ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="causal_v1.0"))
                
            else:
                img = sample_meta["neg_imgs"].pop(0)
                ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="causal_v1.0"))
                

            tmp_list = []
            for img in sample_meta["pos_imgs"]:
                tmp_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"],conv_mode="hypnotized_ans_v1.0"))
            
            for img in sample_meta["neg_imgs"]:
                tmp_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"],conv_mode="hypnotized_ans_v1.0"))
            random.shuffle(tmp_list)
            ret_list = ret_list + tmp_list

        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
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