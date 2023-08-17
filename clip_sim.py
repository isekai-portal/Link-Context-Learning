import sys,os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor,CLIPProcessor,CLIPModel


def read_img_general(img_path):
    return Image.open(img_path).convert('RGB')
    

model = CLIPModel.from_pretrained('/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14')
processor = CLIPProcessor.from_pretrained('/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14')


# vision_tower = '/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'
# image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
# vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
# vision_tower.requires_grad_(False)

print('=================loading complete====================')


img_root = '/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-10/cactus_boxer/pos-cactus_boxer'
#dst = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/tsne/pos.npy'
img_list = os.listdir(img_root)

feat_list = []
for img_item in tqdm(img_list):
    try:
        img_path = os.path.join(img_root,img_item)
        image = read_img_general(img_path)
    except:
        continue

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print('prob: ',probs)
    # image = image_processor.preprocess(image, return_tensors='pt')['pixel_values']

    # image_forward_outs = vision_tower(image, output_hidden_states=True)
    # print('image_forward_outs: ',image_forward_outs.keys())
    # print('image_forward_outs: ',image_forward_outs.last_hidden_state.shape)
    # select_hidden_state_layer = -1
    # select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
    # print('select_hidden_state: ',select_hidden_state.shape)
    # select_hidden_state /= select_hidden_state.norm(dim=-1, keepdim=True)
    # print('select_hidden_state: ',select_hidden_state.shape)
    # similarity = select_hidden_state.cpu().numpy() @ select_hidden_state.cpu().numpy().T
    # print(similarity)
    s
    #feat_list.append(select_hidden_state.reshape(1,-1))
    
#feat_cat = torch.cat(feat_list,dim=0).numpy()


#np.save(dst,feat_cat)