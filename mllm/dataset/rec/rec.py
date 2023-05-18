import sys
import json
import logging
from typing import Dict, Any, List, Sequence

import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from ..common import (
    QuestionTemplateMixin,
    read_img_general,
    BoxDatasetBase,
    Expand2square,
    BoxFormatter,
    AccComputeMetrics,
)

from ..conv import DEFAULT_IMAGE_TOKEN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


# FORMAT
# {
#   "img_path": "zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/images/VG_100K/2350192.jpg",
#   "expression": "this is a table",
#   "bbox": [0.036, 0.68, 0.154, 0.947],  # xyxy
#   "dataset_name": "vgvg",
#   "height": 375,
#   "width": 500
# }

class RECDataset(QuestionTemplateMixin, BoxDatasetBase):
    def __init__(self, *args, data_file, transform=Expand2square(), **kwargs):
        super().__init__(*args, **kwargs)
        self.data_file = data_file
        self.transform = transform

        self.data = []
        with open(data_file, 'r', encoding='utf8') as f:
            for line in tqdm(f, desc='loading annotation file'):
                item: Dict[str, Any] = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def get_raw_item(self, index):
        return self.data[index]

    def get_raw_conv_with_box(self, index) -> List[Dict[str, Any]]:
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']

        img = read_img_general(img_path)
        target = {'bbox': item['bbox']}
        if self.transform is not None:
            img, target = self.transform(img, target)
        bbox = target['bbox']

        question_template: str = self.get_template()
        question = question_template.replace('<expr>', expr) + DEFAULT_IMAGE_TOKEN

        ret = [
            {
                'from': 'human',
                'value': question,
                'image': img,
                'bboxes_seq': None,
            },
            {
                'from': 'gpt',
                'value': 'Answer: <bboxes> .',
                'image': None,
                'bboxes_seq': [[bbox]],
            }
        ]
        return ret


class RECComputeMetrics(AccComputeMetrics):
    def __init__(self, *args, box_formatter: BoxFormatter, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter = box_formatter

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}.")
                continue
            if extract_pred is None:
                failed += 1
                extract_pred = [0, 0, 0, 0]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            ious = box_iou(pred_boxes, target_boxes)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            iou = ious.mean().item()  # please note iou only calculate for success target
            correct = (ious > 0.5).sum().item()

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
