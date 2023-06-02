import os.path
import sys
import json
import logging
import warnings

from PIL import Image
from typing import Dict, Any

from tqdm import tqdm
from torch.utils.data import Dataset

from ..utils import (
    QuestionTemplateMixin,
    read_img_general,
)

from ..root import (
    DATASETS,
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


class ComplexEventBase(QuestionTemplateMixin, Dataset):
    def __init__(self, *args, filename, image_folder=None, subsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        filenames = filename
        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.image_folder = image_folder
        self.subsize = subsize

        self.data = []
        for i, filename in enumerate(filenames):
            # HACK: this annotation broken
            if 'garbage' in filename:
                warnings.warn(f'some annotation may be broken, we just skip it: {filename}')
                continue
            with open(filename, 'r', encoding='utf8') as f:
                data_in_file = []
                for line in tqdm(f, desc=f'loading annotation file {i + 1}/{len(filenames)}'):
                    item: Dict[str, Any] = json.loads(line)
                    data_in_file.append(item)

                data_in_file = self.post_process(data_in_file)

                for item in data_in_file[:self.subsize]:
                    self.data.append(item)
        print(f"dataset size: {len(self.data)}")

    def post_process(self, data_in_file):
        return data_in_file

    def __len__(self):
        return len(self.data)

    def _get_image(self, index):
        item = self.data[index]
        img_path = item['filename']
        if self.image_folder is not None:
            img_path = os.path.join(self.image_folder, img_path)
        image = read_img_general(img_path)
        return image

    def __getitem__(self, index):
        raise NotImplementedError


@DATASETS.register_module()
class ComplexEventCaption(ComplexEventBase):

    def __getitem__(self, index):
        item = self.data[index]
        # image
        image = self._get_image(index)
        # question
        question = self.get_template() + IMAGE_PLACEHOLDER
        # boxes
        boxes = []
        if 'instances' in item and item['instances']:
            for instance in item['instances']:
                boxes.append(instance['bbox'])
        # caption
        if boxes:
            caption = f"Yes. Image Has Target Instances. {BOXES_PLACEHOLDER}"
            boxes_seq = [list(range(len(boxes))), ]
        else:
            caption = "No. Image Has no target instances."
            boxes_seq = []
        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'boxes_seq': boxes_seq,
                }
            ]
        }
        return ret


@DATASETS.register_module()
class ComplexEventGroundCap(ComplexEventBase):
    def post_process(self, data_in_file):
        filtered = []
        for item in data_in_file:
            if 'instances' in item and item['instances']:
                filtered.append(item)
        return filtered

    def __getitem__(self, index):
        item = self.data[index]
        # image
        image = self._get_image(index)
        # question
        question = self.get_template() + IMAGE_PLACEHOLDER
        assert BOXES_PLACEHOLDER in question
        # boxes
        boxes = []
        for instance in item['instances']:
            boxes.append(instance['bbox'])
        # caption
        caption = f"Yes. Image Has Target Instances. {BOXES_PLACEHOLDER}"
        boxes_seq = [list(range(len(boxes))), ]
        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'boxes_seq': boxes_seq,
                }
            ]
        }
        return ret


@DATASETS.register_module()
class ComplexEventREC(ComplexEventBase):
    def post_process(self, data_in_file):
        filtered = []
        for item in data_in_file:
            if 'instances' in item and item['instances']:
                filtered.append(item)
        return filtered

    def __getitem__(self, index):
        item = self.data[index]
        # TODO: use more informative expr
        expr = "object"
        # image
        image = self._get_image(index)
        # question
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr) + IMAGE_PLACEHOLDER
        # boxes
        boxes = []
        for instance in item['instances']:
            boxes.append(instance['bbox'])
        # caption
        caption = f"Yes. Image Has Target Instances. {BOXES_PLACEHOLDER}"
        boxes_seq = [list(range(len(boxes))), ]
        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'boxes_seq': boxes_seq,
                }
            ]
        }
        return ret