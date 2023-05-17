import sys
import json
import logging
from typing import Dict, Any, List

from tqdm import tqdm

from ..common import (
    QuestionTemplateMixin,
    read_img_general,
    BoxDatasetBase,
    Expand2square,
    BoxFormatter,
    AccComputeMetrics,
)

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
        question = question_template.replace('<expr>', expr)

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

    def extract_ans(self, string: str):
        extracted = self.box_formatter.extract(string)
        if len(extracted) <= 0 or len(extracted[0]) <= 0:  # get no boxes
            return None
        return extracted
