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
    PlainBoxFormatter,
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

# FIXME: this version use an annotation file which is preprocessed
#  by expand2square box and plain box formatter
class CaptionBoxDataset(QuestionTemplateMixin, BoxDatasetBase):
    def __init__(self, *args, data_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_file = data_file
        self.transform = Expand2square()
        assert isinstance(self.box_formatter, PlainBoxFormatter)

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
        # image
        img_path = item['image']
        image = read_img_general(img_path)
        # HACK: annotation use expand2square version so we not convert it
        image = self.transform(image)

        # captions
        question_template: str = self.get_template()
        question = question_template + DEFAULT_IMAGE_TOKEN
        # HACK: annotation is converted with box so we not convert it
        caption = item['caption']

        ret = [
            {
                'from': 'human',
                'value': question,
                'image': image,
                'bboxes_seq': None,
            },
            {
                'from': 'gpt',
                'value': caption,
                'image': None,
                'bboxes_seq': None,
            }
        ]
        return ret
