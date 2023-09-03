import imp
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines

from ..utils import MInstrDataset
from .. import BaseComputeMetrics
from typing import Dict, Any, Sequence

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
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def get_neg_pair(self, index, target):
        raise NotImplementedError

    def extract_target(self, string: str):
        try:
            found = string.split("ASSISTANT:")[-1].split("</s>")[0]
            found = found.replace("The answer is", "")
            found = found.replace('there is', '').replace('in the image', '')
            found = found.replace("\"", "").replace("\'", "").replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None

    def extract_pred(self, string: str):
        try:
            found = string.replace("The answer is", "")
            found = found.replace('there is', '').replace('in the image', '')
            found = found.replace("\"", "").replace("\'", "").replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        correct = 0
        failed = 0
        target_failed = 0
        for idx, (pred, target) in enumerate(zip(preds, targets)):
            extract_pred = self.extract_pred(pred)
            extract_target = self.extract_target(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1
            
            pos_target = extract_target
            neg_target = self.get_neg_pair(idx, pos_target)
            if pos_target in pred and neg_target not in pred:
                correct += 1
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }