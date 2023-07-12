# %%
import os
import os.path as osp
import pdb
import json
import csv
from collections import defaultdict, OrderedDict
from tqdm import tqdm
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

anno_file_path = '/mnt/lustre/share_data/zhangzhao2/VG/coco/annotations/instances_train2017.json'
out_tsv_path = '/mnt/lustre/share_data/zhangzhao2/VG/OFA_ceph_annos/ofa_p_det_coco2017.tsv'
ceph_root = 'openmmlab1424:s3://openmmlab/datasets/detection/coco/train2017'

conf_path = '~/petreloss.conf'
client = Client(conf_path)


def read_img_ceph(img_path):
    # img_path= os.path.join(img_path)
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f"Please check image at {img_path}"
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def xywh2xyxy(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def clean_bbox_xyxy(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    if x1 > x2:
        x2 = x1 + 1
    if y1 > y2:
        y2 = y1 + 1

    if x1 < 0:
        x1 = 0.0
    if x2 > img_w:
        x2 = float(img_w)
    if y1 < 0:
        y1 = 0.0
    if y2 > img_h:
        y2 = float(img_h)

    return [x1, y1, x2, y2]


def path_map(src_path, obj_path):

    def inner_map(full_path):
        return full_path.replace(src_path, obj_path)

    return inner_map


# parsing annotation data through COCO API
coco = COCO(anno_file_path)

# %%
cat_ids = coco.getCatIds()

cls_names, text_embeddings, img_embeddings = [], [], []

# get text embeddings
for icat_id in cat_ids:
    cat_name = coco.loadCats(icat_id)[0]['name']
    cat_name = cat_name.replace(',', ' ')
    cls_names.append(cat_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# %%
with torch.no_grad():
    text = clip.tokenize(cls_names).to(device)
    text_features = model.encode_text(text)

# %%
# get image embeddings
nimg = 100
cls_tensors = []
for cat_idx, icat_id in tqdm(enumerate(cat_ids)):
    img_ids = coco.getImgIds(catIds=icat_id)
    selected_ids = sample(img_ids, nimg)
    cls_tensor = torch.zeros([1, text_features.size(1)]).to(device)

    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]

        ann_ids = coco.getAnnIds(imgIds=[img_id],
                                 catIds=[icat_id],
                                 iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        file_name = img_info['file_name']
        ceph_path = osp.join(ceph_root, file_name)

        img = read_img_ceph(ceph_path)
        with torch.no_grad():
            img_tensor = torch.zeros([1, text_features.size(1)]).to(device)
            for ann_idx, iann in enumerate(anns):
                bbox = xywh2xyxy(iann['bbox'])
                bbox = clean_bbox_xyxy(bbox, int(img_info['width']),
                                       int(img_info['height']))
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(
                    bbox[3])
                cropped_img = img[y1:y2, x1:x2]
                try:
                    cropped_img = Image.fromarray(
                        cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                except:
                    cropped_img = Image.fromarray(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cropped_img = preprocess(cropped_img).unsqueeze(0).to(device)
                image_features = model.encode_image(cropped_img)  # [1, 512]
                img_tensor += image_features

        img_tensor = img_tensor / (ann_idx + 1)
        cls_tensor += img_tensor
    cls_tensor = cls_tensor / nimg
    cls_tensors.append(cls_tensor)

cls_features = torch.cat(cls_tensors, dim=0)

# %%
text_features_norm = text_features / torch.norm(
    text_features, dim=-1, keepdim=True)
cls_features_norm = cls_features / torch.norm(
    cls_features, dim=-1, keepdim=True)

i_t_matrix = cls_features_norm.matmul(
    text_features_norm.transpose(1, 0).to(cls_features_norm.dtype))
t_i_matrix = text_features_norm.matmul(
    cls_features_norm.transpose(1, 0).to(text_features_norm.dtype))
i_i_matrix = cls_features_norm.matmul(
    cls_features_norm.transpose(1, 0).to(cls_features_norm.dtype))
t_t_matrix = text_features_norm.matmul(
    text_features_norm.transpose(1, 0).to(text_features_norm.dtype))

# %%
final_matrix = 0.3 * t_t_matrix + 0.5 * (i_t_matrix + t_i_matrix) + i_i_matrix
# final_matrix = t_t_matrix
res, indices = torch.sort(final_matrix, -1, descending=True)

cls_candidates = OrderedDict()
for cat_idx, icat_name in enumerate(cls_names):
    icat_candidates = [cls_names[x] for x in list(indices[cat_idx, :])]
    cls_candidates[icat_name] = icat_candidates

# %%
plt.figure(dpi=300)
sns.heatmap(t_t_matrix.cpu().detach().numpy(), square=True)
plt.title("t_t_matrix")
plt.savefig("./t_t_matrix.png")

plt.figure(dpi=300)
sns.heatmap(i_i_matrix.cpu().detach().numpy(), square=True)
plt.title("i_i_matrix")
plt.savefig("./i_i_matrix.png")

plt.figure(dpi=300)
sns.heatmap(i_t_matrix.cpu().detach().numpy(), square=True)
plt.title("i_t_matrix")
plt.savefig("./i_t_matrix.png")

plt.figure(dpi=300)
sns.heatmap(t_i_matrix.cpu().detach().numpy(), square=True)
plt.title("t_i_matrix")
plt.savefig("./t_i_matrix.png")

plt.figure(dpi=300)
sns.heatmap(final_matrix.cpu().detach().numpy(), square=True)
plt.title("final_matrix")
plt.savefig("./final_matrix.png")

# %%
import csv

with open('./cls_sort.csv', 'w', encoding='utf-8', newline='') as f_csv:
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['cls_name'] + [str(x) for x in range(1, 81)])

    for ican_dic in cls_candidates.items():
        csv_writer.writerow([ican_dic[0]] + ican_dic[1])
