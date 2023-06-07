import os
import os.path as osp
import json
import pdb
import random
from collections import defaultdict
from petrel_client.client import Client
from tqdm import tqdm
import csv
import cv2
from PIL import Image
import numpy as np

conf_path = '~/petreloss.conf'
client = Client(conf_path)

anno_path = '/mnt/lustre/share_data/zhangzhao2/Monolith/academic/PointQA/pointingqa/Datasets/LocalQA/localqa_dataset.json'
split_list_path = '/mnt/lustre/share_data/zhangzhao2/Monolith/academic/PointQA/pointingqa/Datasets/LocalQA/localqa_train_imgs.json'
ceph_root = 'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/'

out_json = 'pointQA_local_train.jsonl'

def load_ceph_json(ceph_path,client):
    if client.contains(ceph_path):
        ceph_json_bytes = client.get(ceph_path)
        ceph_json = json.loads(ceph_json_bytes)
        return ceph_json
    return None

def iter_cpeh_jsonl(path,client):
    response = client.get(path,enable_stream=True,no_cache=True)
    for line in response.iter_lines():
        cur_line = json.loads(line)
        yield cur_line
with open(anno_path,'r') as f_in:
    anno_dict = json.load(f_in)

with open(split_list_path,'r') as f_in:
    split_list = json.load(f_in)


def read_img_general(img_path):
    if "s3://" in img_path:
        cv_img = read_img_ceph(img_path)
        return Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))
    else:
        return Image.open(img_path)


def read_img_ceph(img_path):
    # img_path= os.path.join(img_path)
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f"Please check image at {img_path}"
    img_mem_view = memoryview(img_bytes)
    img_array= np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# convert bbox format, coco2up
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




sub_set1 = set(os.listdir('/mnt/lustre/share_data/zhangzhao2/VG/visual_genome/data/images/VG_100K'))
sub_set2 = set(os.listdir('/mnt/lustre/share_data/zhangzhao2/VG/visual_genome/data/images2/VG_100K_2'))
uniq_id = 0


with open(out_json, 'w', newline='') as f_out:


    for line_idx, img_id in enumerate(split_list):
        if line_idx % 1000 == 0:
            print(f"{line_idx} of {len(split_list)}", flush=True)
        filename = str(img_id) + '.jpg'
        if filename in sub_set1:
            file_rale_path = osp.join('images/VG_100K', filename)
        elif filename in sub_set2:
            file_rale_path = osp.join('images2/VG_100K_2', filename)
        else:
            continue

        anno_info_list = anno_dict[img_id]
        ceph_path = osp.join(ceph_root, file_rale_path)
        if not client.contains(ceph_path):
            print(f"bad path:{ceph_path}", flush=True)
            continue
                
        image = read_img_general(ceph_path).convert("RGB")
        h = image.height
        w = image.width

        for anno_info in anno_info_list:
            ques = anno_info["question"]
            ans_list = anno_info["all_ans"]
            bbox_list = anno_info["all_objs"]
            point_list = anno_info["points"]
            #for ans, point, bbox in zip(ans_list, point_list, bbox_list):
            for i in len(ans_list):
                ans,point,bbox = ans_list[i], point_list[i], bbox_list[i]
                bbox_xywh = [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]
                bbox_xyxy = xywh2xyxy(bbox_xywh)
                bbox_xyxy = clean_bbox_xyxy(bbox_xyxy, w, h)
                point_xy = [point["x"], point["y"]]
                context_info = {
                    "qa_id": uniq_id,
                    "genome_id": img_id,
                    "file_path": file_rale_path,
                    "question": ques,
                    "answer": ans,
                    "bbox": bbox_xyxy,
                    "point": point_xy,
                    "img_h": h,
                    "img_w": w
                }

                f_out.write(json.dumps(context_info, ensure_ascii=False) + '\n')
                f_out.flush()
                uniq_id += 1 