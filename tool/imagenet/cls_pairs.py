import os
import os.path as osp
import pdb
import json
import jsonlines
import csv
from tqdm import tqdm
import numpy as np
import random
import math

neighbor_num = 36
sample_num = 36

dataset_cfg_lst = [{
    'dataset': 'imagenet1k',
    'cls_sort_path':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/similarity_analysis_train900/cls_sort.csv',
    'train_json_path':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/train900.jsonl',
    'test_json_path':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100.jsonl',
    'out_dir':
    '/mnt/lustre/share_data/taiyan/dataset/imagenet1k'
}]


def gen_train_pairs():
    for dataset_info in dataset_cfg_lst:
        id2name = {}
        id2imglist = {}
        json_path = dataset_info['train_json_path']
        with jsonlines.open(json_path) as reader:
            for metas in reader:
                id2name[metas[0]] = metas[1].replace("_"," ")
                id2imglist[metas[0]] = metas[2]

        # load dict
        out_dir = osp.join(dataset_info['out_dir'], 'train900_pairs.jsonl')
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
                bins = math.ceil(num_classes / neighbor_num)
                neighbors = []

                for idx in range(neighbor_num):
                    start = idx * bins
                    end = (idx + 1) * bins
                    if idx == 0:
                        start = 1
                    assert start < num_classes
                    segments = all_cats[start:end]

                    selected_cls_id = random.choice(segments)
                    selected_cls_name = id2name[selected_cls_id]
                    selected_imglist = id2imglist[selected_cls_id]
                    selected_img = random.choice(selected_imglist)
                    neighbor = [selected_cls_id, selected_cls_name, selected_img]
                    neighbors.append(neighbor)

                img_list = id2imglist[cls_id]
                samples = random.sample(img_list, sample_num)

                cls_data = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'samples': samples,
                    'neighbors': neighbors
                }

                writer.write(cls_data)

def gen_test_pairs():
    context_num = 32
    for dataset_info in dataset_cfg_lst:
        json_path = dataset_info['test_json_path']
        out_dir = osp.join(dataset_info['out_dir'], 'test100_pairs.jsonl')

        with jsonlines.open(json_path) as reader, \
            jsonlines.open(out_dir, 'w') as writer:
            for metas in reader:
                cls_id = metas[0]
                cls_name = metas[1].replace("_"," ")
                cls_imgs = metas[2]
                context_samples = random.sample(cls_imgs, context_num)
                test_samples = list(set(cls_imgs).difference(context_samples))

                cls_data = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'context_samples': context_samples,
                    'test_samples': test_samples
                }
                writer.write(cls_data)

if __name__ == '__main__':
    # gen_train_pairs()
    gen_test_pairs()
