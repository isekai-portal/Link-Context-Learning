import sys
import logging
import copy
import json
import os.path
import warnings
from typing import Any, Dict

from tqdm import tqdm
from torch.utils.data import Dataset

from ..root import DATASETS, IMAGE_PLACEHOLDER
from ..utils import read_img_general

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class ConvAnnotationDataset(Dataset):

    def __init__(self, filename, image_folder=None, add_image_placeholder=True):
        self.filename = filename
        self.image_folder = image_folder
        self.add_image_placeholder = add_image_placeholder
        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                obj: Dict[str, Any] = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = copy.deepcopy(self.data[index])

        # image
        if 'image_id' in item:
            image_path = item['image_id']
            if self.image_folder is not None:
                image_path = os.path.join(self.image_folder, image_path)
            image = read_img_general(image_path)
            item['image'] = image
            del item['image_id']

        # target
        if 'boxes' in item:
            item['target'] = {'boxes': item['boxes']}
            del item['boxes']

        # conversations
        if self.add_image_placeholder:
            assert len(item['conversations']) == 2, \
                "only support add image placeholder for 2-round conversation," \
                "please add image placeholder for multi-round conversation in ann file"
            if IMAGE_PLACEHOLDER in item['conversations'][0]['value']:
                warnings.warn(f"already has image_placeholder in item: {item['conversations'][0]['value']}. so we not add twice")
            if 'image' in item and item['image'] is not None:
                item['conversations'][0]['value'] = item['conversations'][0]['value'] + IMAGE_PLACEHOLDER

        return item
