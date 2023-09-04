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

model = CLIPModel.from_pretrained('/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14')
processor = CLIPProcessor.from_pretrained('/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14')


# vision_tower = '/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'
# image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
# vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
# vision_tower.requires_grad_(False)

print('=================loading complete====================')


all_file = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/QR_CODE/all.txt'
positive_file = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/QR_CODE/positive.txt'
img_root = 'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/'
neg_root = 'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/'
prompt = ['a photo of QR code','a photo with no QR code']
des_root = '/mnt/lustre/fanweichen2/tmp_save/output/clip/QR_Code/'
output_file = 'QR_Code.json'
if not os.path.exists(des_root):
    os.makedirs(des_root,exist_ok=True)
positive_list = []
with open(positive_file,'r') as r:
    for line in r.readlines():
        info = json.loads(line)
        filename = info['filename']
        positive_list.append(filename)

with torch.no_grad():
    with open(os.path.join(des_root,output_file),'w') as w:
        with open(all_file,'r') as r:
            for line in tqdm(r.readlines()):
                info = json.loads(line)
                filename = info['filename']
                if filename in positive_list:
                    lable = 'yes'
                else:
                    lable = 'no'
                try:
                    img_path = os.path.join(img_root,filename)
                    value = client.Get(img_path)
                    img = np.frombuffer(value, dtype=np.uint8)
                    image = _common_loader()(img)
                except:
                    img_path = os.path.join(neg_root,filename)
                    value = client.Get(img_path)
                    img = np.frombuffer(value, dtype=np.uint8)
                    image = _common_loader()(img)

                inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1).cpu().numpy()  # we can take the softmax to get the label probabilities
                if probs[0][0] > probs[0][1]:
                    pd = 'yes'
                else:
                    pd = 'no'
                res = {'pd':pd,'gt':lable}
                w.write(json.dumps(res)+'\n')

#dst = '/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/tsne/pos.npy'
# img_list = os.listdir(img_root)

# feat_list = []
# for img_item in tqdm(img_list):
#     try:
#         img_path = os.path.join(img_root,img_item)
#         image = read_img_general(img_path)
#     except:
#         continue

#     inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

#     outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#     probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#     print('prob: ',probs)
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
    #feat_list.append(select_hidden_state.reshape(1,-1))
    
#feat_cat = torch.cat(feat_list,dim=0).numpy()


#np.save(dst,feat_cat)