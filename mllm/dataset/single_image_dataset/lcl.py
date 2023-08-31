import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines

from ..utils import MInstrDataset
from .. import BaseComputeMetrics

from ..root import (
    DATASETS,
    METRICS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)
LABEL_PLACEHOLDER = "<label>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

@DATASETS.register_module()
class LCLDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.data = self._get_annos(self.filename)
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None

    def _get_annos(self, filename):
        cls_metas = []
        with jsonlines.open(filename) as reader:
            for metas in reader:
                cls_metas.append(metas)
        return cls_metas

    def get_raw_item(self, index):
        return self.data[index]

    def get_ret(self, image, question, answer, conv_mode=None):
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"{answer}",
                },
            ]
        }
        if conv_mode is not None:
            ret['mode'] = conv_mode
        return ret

    def get_samples(self, index, mode):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __get_icl_item__(self, index, shot):
        raise NotImplementedError

@METRICS.register_module()
class LCLComputeMetrics(BaseComputeMetrics):
    def extract_ans(self, string: str):
        try:
            found = string.split("ASSISTANT:")[-1].split("</s>")[0].replace("The answer is", "").replace('there is', '').replace('in the image', '').replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None