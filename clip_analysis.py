import sys,os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor


def read_img_general(img_path):
    return Image.open(img_path).convert('RGB')
    
vision_tower = '/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'
image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
vision_tower.requires_grad_(False)

print('=================loading complete====================')


img_root = '/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-10/cactus_boxer/pos-cactus_boxer'
dst = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/tsne/pos.npy'
img_list = os.listdir(img_root)

feat_list = []
for img_item in tqdm(img_list):
    try:
        img_path = os.path.join(img_root,img_item)
        image = read_img_general(img_path)
    except:
        continue
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values']

    image_forward_outs = vision_tower(image, output_hidden_states=True)
    select_hidden_state_layer = -2
    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
    feat_list.append(select_hidden_state.reshape(1,-1))
    
feat_cat = torch.cat(feat_list,dim=0).numpy()


np.save(dst,feat_cat)