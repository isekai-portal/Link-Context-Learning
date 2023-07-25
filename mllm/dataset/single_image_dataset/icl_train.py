import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines
import random

from ..utils import MInstrDataset

from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
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
class ICLTrainDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.data = self._get_annos(self.filename)
        self.cls_neg_label = None

    def _get_annos(self, filename):
        cls_metas = []
        with jsonlines.open(filename) as reader:
            for metas in reader:
                cls_metas.append(metas)
        return cls_metas

    def get_raw_item(self, index):
        return self.data[index]

    def get_template(self):
        raise NotImplementedError 

    def get_ret(self, image, question, answer):
        # Placeholder for template
        # question = item['text']
        # final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {answer}.",
                },
            ]
        }
        return ret

    def get_samples(self, index, mode="cls_positive"):
        assert mode in ['cls_positive','cls_negative', 'neighbors']

        item = self.get_raw_item(index)
        samples = item['samples']
        neighbors = item['neighbors']
        
        if mode == "cls_positive":
            # current class image and label
            label = item['class_name'].lower()
            sample = random.choice(samples)
        elif mode == "cls_negative":
            # current class image, random neighbor label
            if self.cls_neg_label:
                label = self.cls_neg_label
            else:
                metas = random.choice(neighbors)
                label = metas[1].lower()
                self.cls_neg_label = label
            sample = random.choice(samples)
        elif mode == "neighbors":
            # random neighbor image and label
            metas = random.choice(neighbors)
            label = metas[1].lower()
            sample = metas[2]
        else:
            raise NotImplementedError

        image = self.get_image(sample)
        return image, label

    def __getitem__(self, index):
        raise NotImplementedError

    def __get_icl_item__(self, index, shot):
        raise NotImplementedError