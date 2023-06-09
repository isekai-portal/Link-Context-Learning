import json
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
    prediction = json.load(open(prediction_file, 'r', encoding='utf8'))
    preds = [obj['pred'] for obj in prediction['detail']]
    targets = [obj['target'] for obj in prediction['detail']]

    assert len(preds) == len(targets) == len(ds), f"{len(preds)}, {len(ds)}"

    box_formatter = PlainBoxFormatter()

    def extract_ans(string: str):
        list_of_boxes = box_formatter.extract(string)
        return list_of_boxes

    for idx, (pred, target) in enumerate(zip(preds, targets)):
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
    plt.savefig(f'temp.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def main(*args, **kwargs):
    for example in format_examples(*args, **kwargs):
        pprint(example)
        check_example(example)
        _ = input()
        if _ == 'q':
            break


if __name__ == '__main__':
    cfg =dict(
        type='GQADataset',
        image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
        scene_graph_file=None,
        scene_graph_index=None,
        version="q-a",
        template_file=r"/mnt/cache/chenkeqin/unify_mllm/config/_base_/dataset/template/VQA_BCoT.json",
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/gqa_testdev_balanced_questions.jsonl'
    )

    ds = DATASETS.build(cfg=cfg)

    prediction_file = r'/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/llava7_ckpt20000/eval_extra_prediction.json'
    main(
        ds=ds,
        prediction_file=prediction_file,
    )
