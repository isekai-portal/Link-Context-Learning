import json
import os.path
from pprint import pprint

import matplotlib.pyplot as plt

import sys
import pathlib

# sys path append the project dir to enable import from mllm
sys.path.append(f"{pathlib.Path(__file__).parent.parent}")

from mllm.utils import show, draw_bounding_boxes
from mllm.dataset import DATASETS, PlainBoxFormatter, de_norm_box_xyxy, expand2square


def format_examples(
        ds,
        prediction_file,
):
    objs = [json.loads(line) for line in open(prediction_file, 'r', encoding='utf8')]
    preds = [obj['pred'] for obj in objs]
    targets = [obj['target'] for obj in objs]

    assert len(preds) == len(targets) == len(ds), f"{len(preds)}, {len(ds)}"

    box_formatter = PlainBoxFormatter()

    def extract_ans(string: str):
        list_of_boxes = box_formatter.extract(string)
        return list_of_boxes

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        if idx < 72:
            continue
        print(idx)
        raw = ds[idx]
        extract_pred = extract_ans(pred)
        extract_target = extract_ans(target)

        yield dict(
            idx=idx,
            raw=raw,
            extract_pred=extract_pred,
            extract_target=extract_target,
            pred=pred,
            target=target,
        )


def check_example(example):
    idx = example['idx']

    print(example['raw'])
    print(example['pred'])

    image = example['raw']['image']
    boxes_to_draw = [
        example['raw']['target']['boxes'][0],
        [
            example['raw']['target']['points'][0][0] - 2,
            example['raw']['target']['points'][0][1] - 2,
            example['raw']['target']['points'][0][0] + 2,
            example['raw']['target']['points'][0][1] + 2,
        ]
    ]
    color_to_draw = ['red', 'blue']

    try:
        res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=4)
        show(res)
    except:
        print('error when draw box')
    plt.show()
    plt.savefig(f'temp.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def main(*args, **kwargs):
    for example in format_examples(*args, **kwargs):
        # pprint(example)
        check_example(example)
        _ = input()
        if _ == 'q':
            break


if __name__ == '__main__':
    from mmengine import Config

    cfgs = Config.fromfile('config/_base_/dataset/DEFAULT_TEST_POINT_VARIANT.py')
    prediction_file_root = r'/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/llava13b_pretrain8_fc/checkpoint-62000/'

    name = 'POINT_LOCAL_b_val'

    cfg = cfgs['DEFAULT_TEST_POINT_VARIANT'][name]
    prediction_file = os.path.join(prediction_file_root, f'multitest_{name}_extra_prediction.jsonl')

    print(cfg)
    print(prediction_file)

    ds = DATASETS.build(cfg)
    main(
        ds=ds,
        prediction_file=prediction_file,
    )
