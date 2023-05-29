import json
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt

from mllm.dataset import RECDataset, PlainBoxFormatter, de_norm_box_xyxy
from mllm.utils import show, draw_bounding_boxes


def format_examples(
        data_file: str,
        prediction_file: str,
        image_folder=None,
        template_string: str = "Please specify the location of <expr> using the bounding box's top-left and bottom-right coordinates. Make sure the coordinates are normalized between 0 and 1.",
        template_file: Optional[str] = None,
        max_dynamic_size: Optional[str] = None,
):
    rec = RECDataset(filename=data_file,
                     image_folder=image_folder,
                     template_string=template_string,
                     template_file=template_file,
                     max_dynamic_size=max_dynamic_size)

    box_formatter = PlainBoxFormatter()

    def extract_ans(string: str):
        try:
            list_of_boxes = box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            print(f"extract_ans for {string} but get exception: {e}")
            return None

    prediction = json.load(open(prediction_file, 'r', encoding='utf8'))
    preds = [obj['pred'] for obj in prediction['detail']]
    targets = [obj['target'] for obj in prediction['detail']]

    assert len(preds) == len(targets) == len(rec), f"{len(preds)}, {len(rec)}"

    for idx, (pred, target) in enumerate(zip(preds, targets)):
        raw = rec.get_raw_item(idx)
        item = rec.get_raw_conv_with_box(idx)
        extract_pred = extract_ans(pred)
        extract_target = extract_ans(target)

        yield dict(
            raw=raw,
            raw_conv=item,
            extract_pred=extract_pred,
            extract_target=extract_target,
            pred=pred,
            target=target,
        )


def check_example(example):
    raw = example['raw']
    raw_conv = example['raw_conv']
    extract_pred = example['extract_pred']
    if extract_pred is None:
        print('failed to extract pred box')
        extract_pred = [0, 0, 0, 0]
    extract_target = example['extract_target']
    if extract_target is None:
        print('failed to extract target box')
        extract_target = [0, 0, 0, 0]

    image = raw_conv[0]['image']
    ann_box = raw_conv[-1]['bboxes_seq'][0][0]

    bboxes = [de_norm_box_xyxy(_, w=image.width, h=image.height) for _ in (ann_box, extract_pred, extract_target)]
    colors = ['red', 'blue', 'yellow']
    labels = ['gt', 'pred', 'target']
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
    data_file = r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome-tiny/val.jsonl'
    prediction_file = r'/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/rec_ref3_reverse_m8_llava_v1_7b_eval/eval_extra_prediction.json'

    main(
        data_file=data_file,
        prediction_file=prediction_file,
    )
