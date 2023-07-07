import os
import sys
import logging
import pathlib
import typing
import warnings

SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
print(f"SLURM_ENV: {SLURM_ENV}")
project_path = pathlib.Path(__file__).parent.parent
print(f"add project path: `{project_path}` to path to enable import form mllm")
sys.path.append(str(project_path))

import torch
import torch.cuda

from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
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
    cfg, training_args = prepare_args(['config/dummy_llava_train.py', '--overwrite_output_dir'])
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    # Some ugly codes to inject target_processor into preprocessor. maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = smart_prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    print_trainable_params(model)

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)

    # Initialize Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=preprocessor['text'],
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        **data_collator_dict,
    )

    print(dataset['train'])
    print(len(dataset['train']))

    from tqdm import tqdm
    from mllm.utils import decode_generate_ids
    tokenizer = preprocessor['text']
    for idx, item in enumerate(tqdm(dataset['train'])):
        if idx > 10:
            break
        print(item.keys())
        print(decode_generate_ids(tokenizer, item['labels']))
        print(decode_generate_ids(tokenizer, item['input_ids']))
        idx = idx



# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
