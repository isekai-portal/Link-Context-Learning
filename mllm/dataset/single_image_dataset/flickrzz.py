import os.path
import sys
import json
import logging
import warnings
from typing import Dict, Any, Sequence

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.ops import box_iou

from ..utils import (
    QuestionTemplateMixin,
    BaseComputeMetrics,
    read_img_general,
    de_norm_box_xyxy,
    expand2square,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class FlickrZz(QuestionTemplateMixin, Dataset):
    def __init__(self, *args, filename, image_folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.image_folder = image_folder

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for i, line in enumerate(tqdm(f, desc='loading annotations')):
                obj = json.loads(line)
                for j, caption in enumerate(obj['context']):
                    self.data.append(dict(
                        id=i + j,
                        image=obj['img_path'],
                        caption=caption,
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        assert isinstance(index, int), f"Don't know why its type is {type(index)}"  # FIXME
        item = self.data[index]

        # image
        img_path = item['image']
        if self.image_folder is not None:
            img_path = os.path.join(self.image_folder, img_path)
        image = read_img_general(img_path)
        image = expand2square(image)

        # caption
        caption = item['caption']

        # question
        question = self.get_template() + IMAGE_PLACEHOLDER

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                }
            ]
        }
        return ret
