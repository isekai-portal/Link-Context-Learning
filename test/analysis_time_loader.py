import os
import sys
import logging
import pathlib
import time
import warnings

import torch
import torch.cuda

from mllm.config import prepare_args
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, smart_prepare_target_processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def main():
    cfg, training_args = prepare_args()

    from transformers import CLIPImageProcessor, LlamaTokenizer
    from mllm.dataset.process_function import PlainBoxFormatter
    target_processor = {'boxes': PlainBoxFormatter()}
    LLAVA_7B_TK_PATH = r'/test/llava_7b_tk'
    tokenizer = LlamaTokenizer.from_pretrained(LLAVA_7B_TK_PATH)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    preprocessor = dict(
        image=CLIPImageProcessor(),
        text=tokenizer,
        target=target_processor,
        conv=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    st = time.time()
    dl = DataLoader(dataset['train'], batch_size=8,

                    collate_fn=data_collator_dict['train_collator'])
    for i, batch in enumerate(tqdm(dl)):
        pass
    print(f"cost {time.time() - st:.2f} s")


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
