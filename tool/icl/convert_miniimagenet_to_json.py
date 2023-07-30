import json
import os.path
import random
import jsonlines
import tqdm

mappinp_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/devkit/data/map_clsloc.txt'
josn_5way_1shot = '/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/val1000Episode_5_way_1_shot.json'
josn_5way_5shot = '/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/val1000Episode_5_way_5_shot.json'

output_5w1s = '/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/lcl_val1000Episode_5_way_1_shot.jsonl'
output_5w5s = '/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/lcl_val1000Episode_5_way_5_shot.jsonl'

# imagenet1k mapping
folder2name = {}
folder2id = {}
id2name = {}
with open(mappinp_path, 'r') as file:
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    for line in lines:
        info = line.split(' ')
        folder = info[0]
        id = info[1]
        name = info[2]
        if folder not in folder2name.keys():
            folder2name[folder] = name
        if folder not in folder2id.keys():
            folder2id[folder] = id
        if id not in id2name.keys():
            id2name[id] = name

# miniimagenet set
objs = []
with jsonlines.open(josn_5way_5shot) as reader, \
    jsonlines.open(output_5w5s, 'w') as writer:
    for metas in reader:
        for meta in metas:
            support  = meta['Support']
            query = meta['Query']
            folder = [f[0].split('/')[0] for f in support]
            name = [folder2name[f].replace("_"," ") for f in folder]
            writer.write(dict(
                Support = support,
                Query = query,
                Folder = folder,
                Name = name
                )
            )