import os
import sys
import logging
import pathlib
import warnings

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
            trainer.log_metrics("train", train_result.metrics)  # noqa
            trainer.save_metrics("train", train_result.metrics)  # noqa
        except RuntimeError as e:
            with training_args.main_process_first(desc='check cuda device state'):
                for device in range(torch.cuda.device_count()):
                    print(f"#### device {device} summary ####\n{torch.cuda.memory_summary(device)}")
                raise e
        trainer.save_state()  # noqa
        trainer.save_model()
        trainer.plot_loss()

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        warnings.warn(f'try to save cfg to output_dir, but get exception {e}')

    # Keyword arguments for `model.generate`
    gen_kwargs = cfg.data_args.gen_kwargs
    gen_kwargs['pad_token_id'] = preprocessor['text'].pad_token_id
    gen_kwargs['bos_token_id'] = preprocessor['text'].bos_token_id
    gen_kwargs['eos_token_id'] = preprocessor['text'].eos_token_id

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            warnings.warn('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(dataset['validation'], metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_prediction(eval_results, file_key_prefix='eval')

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="test", **gen_kwargs)
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa
        trainer.save_prediction(predict_results, file_key_prefix='test')


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
