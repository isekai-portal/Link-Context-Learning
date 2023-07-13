import sys
import pathlib
import json
from pprint import pprint
from tkinter import N
from typing import Optional

import matplotlib.pyplot as plt

_p = pathlib.Path(__file__).parent.parent
print(_p)
sys.path.append(str(_p))

from mllm.dataset import ReverseRECDataset
from mllm.utils import show, draw_bounding_boxes


def format_examples(
        data_file: str,
        prediction_file: str,
        image_folder=None,
        caption_min_words=None,
        template_string='please generate an unambiguous description for the object <boxes> in the image.',
        template_file: Optional[str] = None,
        max_dynamic_size: Optional[str] = None,
):
    rec = ReverseRECDataset(filename=data_file,
                            image_folder=image_folder,
                            template_string=template_string,
                            caption_min_words=caption_min_words,
                            template_file=template_file,
                            max_dynamic_size=max_dynamic_size)

    prediction = json.load(open(prediction_file, 'r', encoding='utf8'))
    preds = [obj['pred'] for obj in prediction['detail']]
    targets = [obj['target'] for obj in prediction['detail']]

    assert len(preds) == len(targets) == len(rec), f"{len(preds)}, {len(rec)}"

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        raw = rec[idx]

        yield dict(
            raw=raw,
            pred=pred,
            target=target,
        )


def check_example(example):
    image = example['raw']['image']
    bboxes = example['raw']['target']['boxes']
    colors = ['red']
    labels = ['query']
    width = 4
    result = draw_bounding_boxes(image, bboxes, colors=colors, labels=labels, width=width)
    show(result)
    plt.show()
    plt.savefig("./tmp.jpg", dpi=300)
    plt.close()


def main(*args, **kwargs):
    for example in format_examples(*args, **kwargs):
        pprint(example)
        check_example(example)
        _ = input()
        if _ == 'q':
            break


if __name__ == '__main__':
    data_file = r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/val.jsonl'
    prediction_file = r'/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/rec_ref3_reverse_m8_llava_v1_7b_eval_on_flickr_reverse/eval_extra_prediction.json'
    caption_min_words = 8

    main(
        data_file=data_file,
        prediction_file=prediction_file,
        caption_min_words=caption_min_words,
    )
