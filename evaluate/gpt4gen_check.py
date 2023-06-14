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


def format_examples(
        ds,
        tk,
        pred_npy,
        target_npy,
):
    preds = np.load(pred_npy)
    targets = np.load(target_npy)
    preds = decode_generate_ids(tk, preds)
    targets = decode_generate_ids(tk, targets)

    assert len(preds) == len(targets) == len(ds), f"{len(preds)}, {len(ds)}"

    box_formatter = PlainBoxFormatter()

    import re
    pat = re.compile('((?:yes)|(?:no))\.?</s>')

    def extract_ans(string: str):
        list_of_boxes = box_formatter.extract(string)
        return list_of_boxes

    y2n = []
    n2y = []

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        print(idx)
        raw = ds[idx]
        extract_pred = extract_ans(pred)
        extract_target = extract_ans(target)

        # print(pred)
        try:
            extract_pred_ans = pat.findall(pred)[0]
            extract_target_ans = pat.findall(target)[0]
            if extract_pred_ans == extract_target_ans:
                continue
            if extract_pred_ans == 'yes':
                n2y.append(idx)
            else:
                y2n.append(idx)
            yield dict(
                idx=idx,
                raw=raw,
                extract_pred=extract_pred,
                extract_target=extract_target,
                pred=pred,
                target=target,
                pred_ans=extract_pred_ans,
                target_ans=extract_target_ans,
            )
        except:
            pass

    print(len(y2n))
    print(len(n2y))


def check_example(example):
    idx = example['idx']

    image = example['raw']['image']
    image = expand2square(image)
    colors = ['red', 'green', 'blue', 'yellow', '#533c1b', '#c04851']
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(example['extract_pred']):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)

    try:
        res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=4)
        show(res)
    except:
        print('error when draw box')
    plt.show()
    plt.savefig(f'tmp.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def main(*args, **kwargs):
    for example in format_examples(*args, **kwargs):
        pass
        pprint(example)

        check_example(example)
        _ = input()
        if _ == 'q':
            break


if __name__ == '__main__':
    from transformers import AutoTokenizer

    cfg = dict(
        type='GPT4Gen',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/GPT4GEN_BoxCoT_test.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        version='bc',
        template_file=r"/mnt/cache/chenkeqin/unify_mllm/config/_base_/dataset/template/VQA_BCoT.json",
    )
    ds = DATASETS.build(cfg=cfg)

    tk = AutoTokenizer.from_pretrained("/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/llava_pretrain5+rb/checkpoint-38000")

    pred_npy = r'/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/llava_finetune_gpt4gen_qbc_fc/checkpoint-400/eval_predictions.npy'
    target_npy = r'/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/llava_finetune_gpt4gen_qbc_fc/checkpoint-400/eval_label_ids.npy'

    main(
        ds=ds,
        tk=tk,
        pred_npy=pred_npy,
        target_npy=target_npy,
    )
