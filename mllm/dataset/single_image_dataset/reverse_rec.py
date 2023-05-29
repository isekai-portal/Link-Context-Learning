import os.path
import sys
import json
import logging
from typing import Dict, Any

from tqdm import tqdm
from torch.utils.data import Dataset

from ..utils import (
    QuestionTemplateMixin,
    read_img_general,
    de_norm_box_xyxy,
)

from ..root import (
    DATASETS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class ReverseRECDataset(QuestionTemplateMixin, Dataset):
    def __init__(self, *args, filename, image_folder=None, caption_min_words=None, caption_max_words=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.image_folder = image_folder
        self.caption_min_words = caption_min_words
        self.caption_max_words = caption_max_words

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in tqdm(f, desc='loading annotation file'):
                item: Dict[str, Any] = json.loads(line)
                self.data.append(item)

        if self.caption_min_words is not None or self.caption_max_words is not None:
            self.origin_data = self.data
            min_words = self.caption_min_words if self.caption_min_words is not None else 0
            max_words = self.caption_max_words if self.caption_max_words is not None else float('inf')
            self.data = []
            logger.info(f"filter reverse_rec dataset by expression words. items before filter: {len(self.origin_data)}")
            for item in self.origin_data:
                expr_len = len(item['expression'].split())
                if min_words < expr_len < max_words:
                    self.data.append(item)
            logger.info(f"filter reverse_rec dataset by expression words. items after  filter: {len(self.data)}")
        else:
            logger.info(f"dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        if self.image_folder is not None:
            img_path = os.path.join(self.image_folder, img_path)
        image = read_img_general(img_path)
        # for some historical reasons, the box in ann_file is normalized
        bbox = de_norm_box_xyxy(bbox, w=image.width, h=image.height)

        question = self.get_template()
        question = question + IMAGE_PLACEHOLDER
        assert BOXES_PLACEHOLDER in question
        caption = expr

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret
