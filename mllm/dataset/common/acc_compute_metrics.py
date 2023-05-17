import sys
import logging
from copy import deepcopy
from typing import Dict, Any

from transformers import EvalPrediction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class AccComputeMetrics:
    def __init__(self, preprocessor: Dict[str, Any]):
        self.preprocessor = preprocessor
        self.tokenizer = self.preprocessor['text']

    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, Any]:
        eval_preds = deepcopy(eval_preds)  # do not modify origin preds and targets
        preds, targets = eval_preds
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        preds[preds < 0] = self.tokenizer.pad_token_id
        targets[targets < 0] = self.tokenizer.pad_token_id
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        assert len(preds) == len(targets)

        correct = 0
        failed = 0
        target_failed = 0
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1
            if extract_pred == extract_target:
                correct += 1
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }

    def extract_ans(self, string: str):
        raise NotImplementedError
