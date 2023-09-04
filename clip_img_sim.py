import sys,os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor,CLIPProcessor,CLIPModel
from petrel_client.client import Client
import io
import json
from tqdm import tqdm

def read_img_general(img_path):
    return Image.open(img_path).convert('RGB')
    
client = Client()
def _common_loader():
    def pil_loader(img_str):
        buff = io.BytesIO(img_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img
    return pil_loader

def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

vision_tower = '/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'
image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
vision_tower.requires_grad_(False)

print('=================loading complete====================')


img_root = '/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/cactus boxer/pos-cactus boxer'
#dst = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/tsne/pos.npy'
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
    print('image_forward_outs: ',image_forward_outs.keys())
    print('image_forward_outs: ',image_forward_outs.last_hidden_state.shape)
    select_hidden_state_layer = -1
    select_hidden_state = image_forward_outs.last_hidden_state[:,0]
    print('select_hidden_state: ',select_hidden_state.shape)
    cat_b = torch.cat([select_hidden_state,select_hidden_state],dim=0)
    similarity = compute_scores(select_hidden_state,cat_b)
    #similarity = select_hidden_state.cpu().numpy() @ select_hidden_state.cpu().numpy().T
    print(similarity)
    s
    feat_list.append(select_hidden_state.reshape(1,-1))
    
feat_cat = torch.cat(feat_list,dim=0).numpy()

