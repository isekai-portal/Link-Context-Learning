import json
import glob
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt

import sys
import pathlib

# sys path append the project dir to enable import from mllm
sys.path.append(f"{pathlib.Path(__file__).parent.parent}")

from mllm.utils import show, draw_bounding_boxes
from mllm.dataset import DATASETS, PlainBoxFormatter, de_norm_box_xyxy, expand2square


def format_examples(
        cfg,
        prediction_file,
):
    ds = DATASETS.build(cfg=cfg)

    prediction = json.load(open(prediction_file, 'r', encoding='utf8'))
    preds = [obj['pred'] for obj in prediction['detail']]
    targets = [obj['target'] for obj in prediction['detail']]

    assert len(preds) == len(targets) == len(ds), f"{len(preds)}, {len(ds)}"

    box_formatter = PlainBoxFormatter()

    def extract_ans(string: str):
        list_of_boxes = box_formatter.extract(string)
        return list_of_boxes

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        if idx < 4800:
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
    with open(r'samples/captions.jsonl', 'a', encoding='utf8') as g:
        idx = example['idx']
        # g.write(example['pred'])
        # g.write('\n')

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

        from mllm.utils import draw_bounding_boxes, show
        try:
            res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=4)
            show(res)
        except:
            print('error when draw box')
        plt.show()
        # plt.savefig(f'samples/sample_{idx}.jpg', dpi=300, bbox_inches='tight')
        plt.savefig(f'temp3.jpg', dpi=300, bbox_inches='tight')
        plt.close()


def main(*args, **kwargs):
    for example in format_examples(*args, **kwargs):
        pprint(example)
        check_example(example)
        _ = input()
        if _ == 'q':
            break


if __name__ == '__main__':
    cfg = dict(
        type='ComplexEventCaption',
        filename=glob.glob('/mnt/lustre/chenkeqin/share_data/zhangzhao2/Monolith/business/anno/**/*train*.jsonl', recursive=True),
        image_folder=r'/mnt/lustre/chenkeqin/share_data/zhangzhao2/Monolith/business/data',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    )

    prediction_file = r'/mnt/lustre/share_data/chenkeqin/exp_unify_mllm_on_business/ce_cap_llava_flickr/test_extra_prediction.json'

    main(
        cfg=cfg,
        prediction_file=prediction_file,
    )
