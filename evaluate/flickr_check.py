import json
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt

from mllm.dataset import FlickrDataset, PlainBoxFormatter, de_norm_box_xyxy
from mllm.utils import show, draw_bounding_boxes


def format_examples(
        filename: str,
        prediction_file: str,
        image_folder=None,
        template_string: str = "Please specify the location of <expr> using the bounding box's top-left and bottom-right coordinates. Make sure the coordinates are normalized between 0 and 1.",
        template_file: Optional[str] = None,
        max_dynamic_size: Optional[str] = None,
):
    ds = FlickrDataset(filename=filename,
                       image_folder=image_folder,
                       template_string=template_string,
                       template_file=template_file,
                       max_dynamic_size=max_dynamic_size)

    box_formatter = PlainBoxFormatter()

    def extract_ans(string: str):
        list_of_boxes = box_formatter.extract(string)
        return list_of_boxes

    prediction = json.load(open(prediction_file, 'r', encoding='utf8'))
    preds = [obj['pred'] for obj in prediction['detail']]
    targets = [obj['target'] for obj in prediction['detail']]

    assert len(preds) == len(targets) == len(ds), f"{len(preds)}, {len(ds)}"

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        raw = ds[idx]
        extract_pred = extract_ans(pred)
        extract_target = extract_ans(target)

        yield dict(
            raw=raw,
            extract_pred=extract_pred,
            extract_target=extract_target,
            pred=pred,
            target=target,
        )


def check_example(example):
    colors = ['red', 'green', 'blue', 'yellow', '#533c1b', '#c04851']
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(example['extract_pred']):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(box)
            color_to_draw.append(color)

    from mllm.utils import draw_bounding_boxes, show
    res = draw_bounding_boxes(image=example['raw']['image'], boxes=boxes_to_draw, colors=color_to_draw, width=4)
    show(res)
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
    data_file = r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome-tiny/val.jsonl'
    prediction_file = r'/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/eval_rec_llava_origin_5_1/eval_extra_prediction.json'

    main(
        data_file=data_file,
        prediction_file=prediction_file,
    )
