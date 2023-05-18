import sys
import logging
import pathlib

import torch
import torch.cuda

from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def main():
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    print_trainable_params(model)

    # Prepare data_collator
    collator_kwargs = dict(
        padding=cfg.data_args.padding,
        max_length=cfg.data_args.max_length,
    )
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, training_args, preprocessor)

    # Initialize Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=preprocessor['text'],
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['validation'] if training_args.do_eval else None,
        # eval only when use generate_mode
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        **data_collator_dict,
    )

    # Training
    if training_args.do_train:
        try:
            if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                train_result = trainer.train()
        except RuntimeError as e:
            with training_args.main_process_first(desc='check cuda device state'):
                for device in range(torch.cuda.device_count()):
                    print(f"#### device {device} summary ####\n{torch.cuda.memory_summary(device)}")
                raise e
        trainer.log_metrics("train", train_result.metrics)  # noqa
        trainer.save_metrics("train", train_result.metrics)  # noqa
        trainer.save_state()  # noqa
        trainer.save_model()
        trainer.plot_loss()

    # Keyword arguments for `model.generate`
    gen_kwargs = cfg.data_args.gen_kwargs
    gen_kwargs['pad_token_id'] = preprocessor['text'].pad_token_id
    gen_kwargs['bos_token_id'] = preprocessor['text'].bos_token_id
    gen_kwargs['eos_token_id'] = preprocessor['text'].eos_token_id

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)  # noqa
        trainer.save_metrics("eval", metrics)  # noqa

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="predict", **gen_kwargs)
        trainer.save_predict(predict_results)


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
