import sys
import logging

from typing import Any, Dict, Union, List

import torch
from torch import nn

from .base_engine import TrainerForMMLLM, Seq2Seq2DataCollatorWithImage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

IGNORE_INDEX = -100


class TrainerForOpenFlamingo(TrainerForMMLLM):

    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer is not None, ""
        self.media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        self.endofchunk_token_id = self.tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        if '<answer>' in self.tokenizer.get_vocab():
            self.answer_token_id = self.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        else:
            self.answer_token_id = None

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        label_ids = inputs['labels']
        label_ids[label_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX
        label_ids[label_ids == self.tokenizer.bos_token_id] = IGNORE_INDEX
        label_ids[label_ids == self.media_token_id] = IGNORE_INDEX
        if self.answer_token_id is not None:
            for i in range(label_ids.shape[0]):
                # remove loss for any token before <answer> token
                label_idx = 0
                while (
                        label_idx < label_ids.shape[1] and label_ids[i][label_idx] != self.answer_token_id
                ):
                    label_ids[i][label_idx] = -100
                    label_idx += 1
            label_ids[label_ids == self.answer_token_id] = IGNORE_INDEX
        inputs['labels'] = label_ids

        # noinspection PyUnresolvedReferences
        output = super().training_step(model, inputs)

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[self.media_token_id] = torch.ones_like(zero_mask[self.media_token_id])
                zero_mask[self.endofchunk_token_id] = torch.ones_like(zero_mask[self.endofchunk_token_id])
                if self.answer_token_id is not None:
                    zero_mask[self.answer_token_id] = torch.ones_like(zero_mask[self.answer_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)
        return output


class DataCollatorForOpenFlamingo(Seq2Seq2DataCollatorWithImage):
    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = super()._image_process(features)
        if ret['images'].ndim == 4:
            # B C H W -> B N T C H W
            ret['images'] = ret['images'].unsqueeze(1).unsqueeze(1)
        assert ret['images'].ndim == 6
        return ret
