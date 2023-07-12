import os
import os.path as osp
import pdb
import json
import jsonlines
import csv
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
import random
import math

neg_cls_num = 36
pos_sample_num = 36

dataset_cfg_lst = [{
    'dataset': 'imagenet1k',
    'json_path':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/imagenet1k.jsonl',
    'cls_sort_path':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/similarity_analysis/cls_sort.csv'
}]
out_dir = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/imagenet1k_pairs.jsonl'


for dataset_info in dataset_cfg_lst:

    id2name = {}
    id2imglist = {}
    # cls_ids, cls_names, cls_imgs = [], [], []
    json_path = dataset_info['json_path']
    with jsonlines.open(json_path) as reader:
        for metas in reader:
            # cls_ids.append(metas[0])
            # cls_names.append(metas[1])
            # cls_imgs.append(metas[2])
            id2name[metas[0]] = metas[1]
            id2imglist[metas[0]] = metas[2]

    outputs = []
    # load dict
    sort_cls_dict = {}
    with open(dataset_info['cls_sort_path'],
              'r') as f_sort, jsonlines.open(out_dir, 'w') as writer:
        tsv_reader = csv.reader(f_sort, delimiter=',')
        for line_idx, line in enumerate(tsv_reader):
            if line_idx == 0:  # skip head
                continue

            cls_id = line[0].strip()
            cls_name = id2name[cls_id]
            all_cats = [x.strip() for x in line[1:]]
            # remove self for each class
            num_classes = len(all_cats)
            bins = math.ceil(num_classes / neg_cls_num)
            neg_samples = []

            for idx in range(neg_cls_num):
                start = idx * bins
                end = (idx + 1) * bins
                if idx == 0:
                    start = 1
                assert start < num_classes
                segments = all_cats[start:end]

                selected_cls_id = random.sample(segments, 1)[0]
                selected_cls_name = id2name[selected_cls_id]
                selected_imglist = id2imglist[selected_cls_id]
                selected_img = random.sample(selected_imglist, 1)[0]
                neg_sample = [selected_cls_id, selected_cls_name, selected_img]
                neg_samples.append(neg_sample)

            pos_imglist = id2imglist[cls_id]
            pos_samples = random.sample(pos_imglist, pos_sample_num)

            cls_data = {
                'class_id': cls_id,
                'class_name': cls_name,
                'positive_samples': pos_samples,
                'negative_samples': neg_samples
            }

            writer.write(cls_data)
