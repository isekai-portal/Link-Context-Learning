import json
from pprint import pprint

import matplotlib.pyplot as plt

import sys
import pathlib
import numpy as np
from regex import P

# sys path append the project dir to enable import from mllm
sys.path.append(f"{pathlib.Path(__file__).parent.parent}")

from mllm.utils import show, draw_bounding_boxes, decode_generate_ids
from mllm.dataset import DATASETS, PlainBoxFormatter, de_norm_box_xyxy, expand2square


if __name__ == '__main__':
    from transformers import AutoTokenizer

    cfg = dict(
        type='GPT4Gen',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/GPT4GEN_BoxCoT_train.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        version='bc',
        template_file=r"/mnt/cache/chenkeqin/unify_mllm/config/_base_/dataset/template/VQA_BCoT.json",
    )
    ds = DATASETS.build(cfg=cfg)

    for idx in range(len(ds)):
        item = ds[idx]
        print(item)

        image = item['image']
        colors = ['red', 'green', 'blue', 'yellow', '#533c1b', '#c04851']
        boxes_to_draw = []
        color_to_draw = []

        all_boxes = item['target']['boxes']
        for idx, boxes_idx in enumerate(item['conversations'][-1]['boxes_seq']):
            color = colors[idx % len(colors)]
            for box_idx in boxes_idx:
                boxes_to_draw.append(all_boxes[box_idx])
                color_to_draw.append(color)
        try:
            res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=4)
            show(res)
        except:
            print('error when draw box')
        plt.show()
        plt.savefig(f'tmp.jpg', dpi=300, bbox_inches='tight')
        plt.close()

        _ = input()
        if _ == 'q':
            break