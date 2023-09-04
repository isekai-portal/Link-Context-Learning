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
from mllm.dataset.utils.io import read_img_general

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor,CLIPProcessor,CLIPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def image_crop(img,bbox):
    crop_img = img.crop((int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
    return crop_img

@DATASETS.register_module()
class BusinessVQADataset_MIX(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.data = self._get_annos(self.filename)
        self.data_positive = self._get_annos(self.filename_positive)
        self.data_negative = self._get_annos(self.filename_negative)

        self.cls_neg_label = None
        self.support_positive_embedding = None
        self.support_negative_embedding = None
        self.positive_done = False
        self.negative_done = False

        self.vision_tower = '/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower).cuda()
        self.vision_tower.requires_grad_(False)

    def _get_annos(self, filename):
        cls_metas = []
        with jsonlines.open(filename) as reader:
            for metas in reader:
                cls_metas.append(metas)
        return cls_metas

    def get_raw_item(self, index, is_icl,mode=None,false_mode=False):
        if not false_mode:
            if not is_icl:
                return self.data[index]
            else:
                if mode == 'cls_positive':
                    return self.data_positive[index]
                else:
                    return self.data_negative[index]
        else:
            return self.data[index]
        # if not is_icl:
        #     return self.data[index]
        # else:
        #     if mode == 'cls_positive':
        #         return self.data_positive[index]
        #     else:
        #         return self.data_negative[index]


    def get_template(self):
        return 

    def get_image(self, image_folder, image_path):
        if image_folder is not None:
            image_path = os.path.join(image_folder, image_path)
        image = read_img_general(image_path)
        return image

    def _get_ret_origin(self, index, mode):


        item = self.get_raw_item(index,is_icl=False)
        try:
            img_id = item['filename']
        except:
            img_id = item['image_id']
        

        Q3 = 'Is there any <exp> in the image <image> ?'
        final_question = Q3
        if self.label_name is not None:
            label = self.label_name

        final_question = final_question.replace('<exp>',label)

        if mode == 'cls_positive':
            image = self.get_image(self.image_folder_positive,img_id)
        else:
            image = self.get_image(self.image_folder_negative,img_id)
        try:
            box = item['boxes'][0]
            if len(box) != 0:
                image = image_crop(image,box)
        except:
            pass

        if mode == 'cls_positive':
            label = 'yes'
        else:
            label = 'no'

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label}.",
                },
            ]
        }
        return ret
    
    def __getitem__(self, index):
        if self.data[index] in self.data_positive:
            return self._get_ret_origin(index, mode = "cls_positive")
        else:
            return self._get_ret_origin(index, mode = "cls_negative")
            
    @torch.no_grad()
    def __get_icl_item__(self, index, shot):
        ret_list = []


        # if self.data[index] in self.data_positive:
        #     query_item = self.get_raw_item(index,is_icl=False,mode="cls_positive",false_mode=False)
        #     query_mode = "cls_positive"
        #     try:
        #         img_id = query_item['filename']
        #     except:
        #         img_id = query_item['image_id']
        #     query_image = self.get_image(self.image_folder_positive,img_id)

        #     image = self.image_processor.preprocess(query_image, return_tensors='pt')['pixel_values']
        #     image_forward_outs = self.vision_tower(image.cuda(), output_hidden_states=True)
        #     select_hidden_state = image_forward_outs.last_hidden_state[:,0]
        #     query_embedding = image_forward_outs.last_hidden_state[:,0].cpu()

        # else:
        #     query_item = self.get_raw_item(index,is_icl=False,mode="cls_negative",false_mode=False)
        #     query_mode = "cls_positive"
        #     try:
        #         img_id = query_item['filename']
        #     except:
        #         img_id = query_item['image_id']
        #     query_image = self.get_image(self.image_folder_negative,img_id)

        #     image = self.image_processor.preprocess(query_image, return_tensors='pt')['pixel_values']
        #     image_forward_outs = self.vision_tower(image.cuda(), output_hidden_states=True)
        #     select_hidden_state = image_forward_outs.last_hidden_state[:,0]
        #     query_embedding = image_forward_outs.last_hidden_state[:,0].cpu()

        
        # if query_mode == 'cls_positive':
        #     if not self.positive_done:
        #         for support_item in self.data_positive:
        #             try:
        #                 img_id = support_item['filename']
        #             except:
        #                 img_id = support_item['image_id']
        #             image = self.get_image(self.image_folder_positive,img_id)
        #             image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        #             image_forward_outs = self.vision_tower(image.cuda(), output_hidden_states=True)
        #             select_hidden_state = image_forward_outs.last_hidden_state[:,0].cpu()
        #             if self.support_positive_embedding is None:
        #                 self.support_positive_embedding = select_hidden_state
        #             else:
        #                 self.support_positive_embedding = torch.cat([self.support_positive_embedding,select_hidden_state],dim=0)
        #         self.positive_done = True
        #     if self.data[index] in self.data_positive:
        #         bad_idx = index
        #     else:
        #         bad_idx = 0
        #     embedding = torch.cat([self.support_positive_embedding[:bad_idx],self.support_positive_embedding[bad_idx:]])
        #     similarity = compute_scores(query_embedding,embedding)
        #     similarity = torch.tensor(similarity)
        #     _,top_idx = torch.topk(similarity,shot)

            
        #     for i in range(len(top_idx)):
        #         simi_idx = top_idx[i].item()
        #         if i == 0:
        #             ret_list.append(self._get_ret(simi_idx, mode = "cls_positive",conv_mode="causal_v1.0"))
        #         else:
        #             ret_list.append(self._get_ret(simi_idx, mode = "cls_positive",conv_mode="hypnotized_ans_v1.0"))

        #     tmp_idx = 0
        #     for _ in range(shot):
        #         ret_list.append(self._get_ret(tmp_idx, mode = "cls_negative",conv_mode="hypnotized_ans_v1.0"))
        #         tmp_idx += 1
        # else:
        #     if not self.negative_done:
        #         for support_item in self.data_negative:

        #             try:
        #                 img_id = support_item['filename']
        #             except:
        #                 img_id = support_item['image_id']
        #             image = self.get_image(self.image_folder_negative,img_id)
        #             image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        #             image_forward_outs = self.vision_tower(image.cuda(), output_hidden_states=True)
        #             select_hidden_state = image_forward_outs.last_hidden_state[:,0].cpu()
        #             if self.support_negative_embedding is None:
        #                 self.support_negative_embedding = select_hidden_state
        #             else:
        #                 self.support_negative_embedding = torch.cat([self.support_negative_embedding,select_hidden_state],dim=0)
        #         self.negative_done = True
        #     if self.data[index] in self.data_negative:
        #         bad_idx = index
        #     else:
        #         bad_idx = 0
        #     embedding = torch.cat([self.support_negative_embedding[:bad_idx],self.support_negative_embedding[bad_idx:]])
        #     similarity = compute_scores(query_embedding,embedding)
        #     similarity = torch.tensor(similarity)
        #     _,top_idx = torch.topk(similarity,shot)

            
        #     for i in range(len(top_idx)):
        #         simi_idx = top_idx[i].item()
        #         if i == 0:
        #             ret_list.append(self._get_ret(simi_idx, mode = "cls_negative",conv_mode="causal_v1.0"))
        #         else:
        #             ret_list.append(self._get_ret(simi_idx, mode = "cls_negative",conv_mode="hypnotized_ans_v1.0"))

        #     tmp_idx = 0
        #     for _ in range(shot):
        #         ret_list.append(self._get_ret(tmp_idx, mode = "cls_positive",conv_mode="hypnotized_ans_v1.0"))
        #         tmp_idx += 1


        tmp_idx = 0
        for i in range(shot):
            if i == 0:
                ret_list.append(self._get_ret(tmp_idx, mode = "cls_positive",conv_mode="causal_v1.0"))
            else:
                ret_list.append(self._get_ret(tmp_idx, mode = "cls_positive",conv_mode="hypnotized_ans_v1.0"))
            tmp_idx += 1

        tmp_idx = 0
        for _ in range(shot):
            ret_list.append(self._get_ret(tmp_idx, mode = "cls_negative",conv_mode="hypnotized_ans_v1.0"))
            tmp_idx += 1

        if self.data[index] in self.data_positive:
            ret_list.append(self._get_ret(index, mode = "cls_positive",is_icl=False,conv_mode="final_v1.0"))
        else:
            ret_list.append(self._get_ret(index, mode = "cls_negative",is_icl=False,conv_mode="final_v1.0"))

        return ret_list


    def _get_ret(self, index, mode="cls_positive", is_icl=True, false_mode=False, conv_mode=None):
        assert mode in ['cls_positive','cls_negative']

        item = self.get_raw_item(index,is_icl,mode,false_mode=false_mode)
        try:
            img_id = item['filename']
        except:
            img_id = item['image_id']
        
        normal_question = '[BEGIN EXAMPLE] What is in the image <image> ?'
        bind_question = 'What is in the image <image> ?'

        if self.label_name is not None:
            label = self.label_name
        if self.label_negative is not None:
            label_negative = self.label_negative
        else:
            label_negative = 'no_'+label
        Q3 = normal_question
        Q3 = Q3.replace('<exp>',label)
        #bind_question = 'Is there any <exp> in the image <image> ?'
        bind_question = bind_question.replace('<exp>',label)
        if not is_icl:
            final_question = bind_question
            #image = self.get_image(self.image_folder,img_id)
            #conv_mode = 'icl'
            if mode == 'cls_positive':
                image = self.get_image(self.image_folder_positive,img_id)
                label = 'yes'
            else:
                image = self.get_image(self.image_folder_negative,img_id)
                label = 'no'
        else:
            if mode == 'cls_positive':
                final_question = Q3
                image = self.get_image(self.image_folder_positive,img_id)
                #label = label+'.'
                label = f'there is "{label}" in the image. [END EXAMPLE]'
            else:
                final_question = Q3
                image = self.get_image(self.image_folder_negative,img_id)
                label = f'there is "{label_negative}" in the image. [END EXAMPLE]'
                #label = label_negative+'.'
            #conv_mode = 'icl'

        # try:
        #     box = item['bbox'][0]
        #     if len(box) != 0:
        #         image = image_crop(image,box)
        # except:
        #     pass
        
        if not is_icl:
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                    },
                    {
                        'from': 'gpt',
                        'value': f"{label}",
                    },],
                'mode': conv_mode
            }
        else:
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': final_question,
                    },
                    {
                        'from': 'gpt',
                        'value': f"{label}",
                    },],
                'mode': conv_mode
            }

        return ret
