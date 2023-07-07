import sys
import logging
from typing import Any, Dict, Union

import torch
from torch import nn

from .base_engine import TrainerForMMLLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

IGNORE_INDEX = -100


class OtterTrainer(TrainerForMMLLM):

    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer is not None, ""
        self.media_token_id = self.tokenizer.get_vocab()["<image>"]
        self.endofchunk_token_id = self.tokenizer.get_vocab()["<|endofchunk|>"]
        self.answer_token_id = self.tokenizer.get_vocab()["<answer>"]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # noinspection PyUnresolvedReferences
        output = super().training_step(model, inputs)

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[self.media_token_id] = torch.ones_like(zero_mask[self.media_token_id])
                zero_mask[self.endofchunk_token_id] = torch.ones_like(zero_mask[self.endofchunk_token_id])
                zero_mask[self.answer_token_id] = torch.ones_like(zero_mask[self.answer_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)
        return output
