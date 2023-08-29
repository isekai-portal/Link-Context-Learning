# %%
import os
import os.path as osp
import pdb
import json
import jsonlines
import csv
from collections import defaultdict, OrderedDict
from tqdm import trange, tqdm
import re
import cv2
import numpy as np
import torch.nn.functional as F

import torch
import clip
from PIL import Image
from random import sample

import seaborn as sns
import matplotlib.pyplot as plt
import csv

from petrel_client.client import Client
# import debugpy;debugpy.connect(('10.142.4.32', 5610))

Dataset = "1k"
if Dataset == "1k":
    ceph_root = 'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/'
    anno_file_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/train900.jsonl'
    output_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/similarity_analysis_train900'
elif Dataset == "22k":
    ceph_root = 'openmmlab1984:s3://openmmlab/datasets/classification/imagenet22k/train/'
    anno_file_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/icl/imagenet22k_train.jsonl'
    output_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/icl/similarity_analysis/'

if not osp.exists(output_path):
    os.mkdir(output_path)

conf_path = '~/petreloss.conf'
client = Client(conf_path)

def read_img_ceph(img_path):
    try:
        img_bytes = client.get(img_path)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        # noinspection PyUnresolvedReferences
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        print("Broken file:{}".format(img_path))
        return None
    return img

cls_ids, cls_names, cls_imgs, text_embeddings, img_embeddings = [], [], [], [], []

# get text embeddings
with jsonlines.open(anno_file_path) as reader:
    for metas in reader:
        cls_ids.append(metas[0])
        if isinstance(metas[1], list):
            cls_names.append(metas[1][0])
        else:
            cls_names.append(metas[1])
        cls_imgs.append(metas[2])

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.eval()

# # debug
# cls_ids = cls_ids[:100]
# cls_names = cls_names[:100]
# cls_imgs = cls_imgs[:100]

with torch.no_grad():
    text = clip.tokenize(cls_names).to(device)
    text_features = model.encode_text(text)

# get image embeddings
nimg = 100
cls_tensors = []
num_cats = len(cls_ids)

tbar = trange(num_cats)
for idx, (cls_id, cls_name, cls_img) in enumerate(zip(cls_ids, cls_names, cls_imgs)):
    if len(cls_img)<nimg:
        selected_imgs=cls_img
    else:
        selected_imgs = sample(cls_img,nimg)
    cls_tensor = torch.zeros([1, text_features.size(1)]).to(device)

    broken_count = 0
    for img_path in tqdm(selected_imgs):
        ceph_path = osp.join(ceph_root, img_path)
        img = read_img_ceph(ceph_path)
        if img is None:
            broken_count +=1
            continue
        with torch.no_grad():
            img_tensor = torch.zeros([1, text_features.size(1)]).to(device)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = preprocess(img).unsqueeze(0).to(device)
            image_features = model.encode_image(img)  # [1, 512]
            img_tensor += image_features

        cls_tensor += img_tensor
    cls_tensor = cls_tensor / (len(selected_imgs) - broken_count)
    cls_tensors.append(cls_tensor)
    tbar.set_description('Class: {}/{}'.format(idx + 1, num_cats))
    tbar.update()

with torch.no_grad():
    cls_features = torch.cat(cls_tensors, dim=0)

    text_features_norm = text_features / torch.norm(text_features, dim=-1, keepdim=True)
    cls_features_norm = cls_features / torch.norm(cls_features, dim=-1, keepdim=True)

    print('Matrix multiplication calculation started...', flush=True)
    i_t_matrix = cls_features_norm.matmul(text_features_norm.transpose(1, 0).to(cls_features_norm.dtype))
    print('Matrix multiplication calculation started...', flush=True)
    t_i_matrix = text_features_norm.matmul(cls_features_norm.transpose(1, 0).to(text_features_norm.dtype))
    print('Matrix multiplication calculation started...', flush=True)
    i_i_matrix = cls_features_norm.matmul(cls_features_norm.transpose(1, 0).to(cls_features_norm.dtype))
    print('Matrix multiplication calculation started...', flush=True)
    t_t_matrix = text_features_norm.matmul(text_features_norm.transpose(1, 0).to(text_features_norm.dtype))

    final_matrix = 0.5 * t_t_matrix + 0.8 * (i_t_matrix + t_i_matrix) + 0.5 * i_i_matrix

    torch.save(i_t_matrix, osp.join(output_path,'i_t_matrix.pth'))
    torch.save(t_i_matrix, osp.join(output_path,'t_i_matrix.pth'))
    torch.save(i_i_matrix, osp.join(output_path,'i_i_matrix.pth'))
    torch.save(t_t_matrix, osp.join(output_path,'t_t_matrix.pth'))

    res, indices = torch.sort(final_matrix, -1, descending=True)

    cls_candidates = OrderedDict()
    for cat_idx, cls_id in enumerate(cls_ids):
        icat_candidates = [cls_ids[x] for x in list(indices[cat_idx, :])]
        cls_candidates[cls_id] = icat_candidates

    # %%
    plt.figure(dpi=300)
    sns.heatmap(t_t_matrix.cpu().detach().numpy(), square=True)
    plt.title("t_t_matrix")
    plt.savefig(osp.join(output_path,'t_t_matrix.png'))

    plt.figure(dpi=300)
    sns.heatmap(i_i_matrix.cpu().detach().numpy(), square=True)
    plt.title("i_i_matrix")
    plt.savefig(osp.join(output_path,'i_i_matrix.png'))

    plt.figure(dpi=300)
    sns.heatmap(i_t_matrix.cpu().detach().numpy(), square=True)
    plt.title("i_t_matrix")
    plt.savefig(osp.join(output_path,'i_t_matrix.png'))

    plt.figure(dpi=300)
    sns.heatmap(t_i_matrix.cpu().detach().numpy(), square=True)
    plt.title("t_i_matrix")
    plt.savefig(osp.join(output_path,'t_i_matrix.png'))

    plt.figure(dpi=300)
    sns.heatmap(final_matrix.cpu().detach().numpy(), square=True)
    plt.title("final_matrix")
    plt.savefig(osp.join(output_path,'final_matrix.png'))

    # %%
    import csv

    with open(osp.join(output_path,'cls_sort.csv'), 'w', encoding='utf-8', newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(['cls_id\sort_idx'] + [str(x) for x in range(1, 81)])

        for ican_dic in cls_candidates.items():
            csv_writer.writerow([ican_dic[0]] + ican_dic[1])
