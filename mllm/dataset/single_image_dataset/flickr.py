import copy
import random
import re
import sys
import json
import logging
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Sequence, List

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from ..root import DATASETS, BOXES_PLACEHOLDER, IMAGE_PLACEHOLDER
from ..utils import QuestionTemplateMixin, read_img_general
from ..utils.flickr30k_entities_utils import (
    flatten_annotation,
    get_img_path,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

NO_BOXES_PLACEHOLDER = ""


class FlickrParser(Dataset):
    def __init__(self, filename, annotation_dir):
        self.filename = filename
        self.annotation_dir = annotation_dir

        self.indexes = [line.strip() for line in open(filename, 'r', encoding='utf8')]
        self.data = flatten_annotation(self.annotation_dir, self.indexes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dump(self, filename):
        import json
        with open(filename, 'w', encoding='utf8') as f:
            for obj in self.data:
                obj_str = json.dumps(obj)
                f.write(obj_str)
                f.write('\n')


@DATASETS.register_module()
class FlickrDataset(QuestionTemplateMixin, Dataset):

    def __init__(self, *args, filename, image_folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.image_folder = image_folder if image_folder is not None else ""

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in tqdm(f, desc='loading annotation file'):
                item: Dict[str, Any] = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # image
        img_path = get_img_path(item['image_id'], image_dir=self.image_folder)
        image = read_img_general(img_path)
        # caption
        caption: str = item['sentence']
        caption = caption.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        # question
        question = self.get_template() + IMAGE_PLACEHOLDER
        ret = {
            'image': image,
            'target': {'boxes': copy.deepcopy(item['boxes'])},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'boxes_seq': copy.deepcopy(item['boxes_seq']),
                }
            ]
        }
        return ret


@DATASETS.register_module()
class FlickrBox2Caption(QuestionTemplateMixin, Dataset):

    def __init__(self, *args, filename, box_max_num=5, caption_with_box='none', image_folder=None, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        assert caption_with_box in ['none', 'all', 'question']
        self.filename = filename
        self.box_max_num = box_max_num
        self.caption_with_box = caption_with_box
        self.image_folder = image_folder if image_folder is not None else ""
        self.seed = seed
        self.rng_state = None
        with self.rng_state_context():
            random.seed(seed)
        assert self.rng_state is not None

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in tqdm(f, desc='loading annotation file'):
                item: Dict[str, Any] = json.loads(line)
                self.data.append(item)

    @contextmanager
    def rng_state_context(self):
        old_state = random.getstate()
        if self.rng_state is not None:
            random.setstate(self.rng_state)
        yield
        self.rng_state = random.getstate()
        random.setstate(old_state)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with self.rng_state_context():
            item = self.data[index]
            # image
            img_path = get_img_path(item['image_id'], image_dir=self.image_folder)
            image = read_img_general(img_path)

            boxes_seq_origin = item['boxes_seq']
            caption_origin: str = item['sentence']
            question_origin: str = self.get_template() + IMAGE_PLACEHOLDER
            assert BOXES_PLACEHOLDER in question_origin

            # select
            if len(boxes_seq_origin) <= self.box_max_num:
                select_boxes_index_x = list(range(len(boxes_seq_origin)))
            else:
                select_boxes_index_x = random.sample(list(range(len(boxes_seq_origin))), k=self.box_max_num)
                select_boxes_index_x.sort()
            select_boxes_index_y = []
            for _x in select_boxes_index_x:
                select_boxes_index_y.append(random.choice(list(range(len(boxes_seq_origin[_x])))))

            # query
            boxes_seq_query: List[int] = []
            for _x, _y in zip(select_boxes_index_x, select_boxes_index_y):
                boxes_seq_query.append(boxes_seq_origin[_x][_y])
            boxes_seq_query = list(set(boxes_seq_query))
            random.shuffle(boxes_seq_query)

            if self.caption_with_box == 'none':
                question = self.get_template() + "The generated caption should only contain words, no boxes." + IMAGE_PLACEHOLDER
            elif self.caption_with_box == 'all':
                question = self.get_template() + IMAGE_PLACEHOLDER
            elif self.caption_with_box == 'question':
                question = self.get_template() + "The generated caption should contain and only contain the query boxes." + IMAGE_PLACEHOLDER
            else:
                assert False

            # answer
            if self.caption_with_box == 'none':
                boxes_seq_answer = []
                caption = caption_origin.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, "")
            elif self.caption_with_box == 'all':
                boxes_seq_answer = copy.deepcopy(boxes_seq_origin)
                caption = caption_origin
            elif self.caption_with_box == 'question':
                boxes_seq_answer = []
                for _x, _y in zip(select_boxes_index_x, select_boxes_index_y):
                    boxes_seq_answer.append([boxes_seq_origin[_x][_y]])
                caption = caption_origin.replace(PHRASE_ST_PLACEHOLDER, "")
                boxes_strs = [NO_BOXES_PLACEHOLDER for _ in range(len(boxes_seq_origin))]
                for _x in select_boxes_index_x:
                    boxes_strs[_x] = BOXES_PLACEHOLDER
                caption = caption.replace(PHRASE_ED_PLACEHOLDER, '{}').format(*boxes_strs)
            else:
                assert False

            ret = {
                'image': image,
                'target': {'boxes': copy.deepcopy(item['boxes'])},
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                        'boxes_seq': [boxes_seq_query],
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                        'boxes_seq': boxes_seq_answer,
                    }
                ]
            }
        return ret
