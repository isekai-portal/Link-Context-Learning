import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINER_STATE_NAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class TrainerDifferentCollatorMixin:
    def __init__(self,
                 *args,
                 train_collator: Optional[DataCollator] = None,
                 eval_collator: Optional[DataCollator] = None,
                 test_collator: Optional[DataCollator] = None,
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
        self._test_collator = test_collator if test_collator is not None else self._eval_collator
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_train_dataloader(self) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader


# noinspection DuplicatedCode
class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom behavior.

        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs:
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys())
        with torch.no_grad():
            with self.compute_loss_context_manager():
                generated_tokens = self.model.generate(**gen_kwargs)

        # important for Decoder-Only LLM: only extract generated_tokens and discard origin inputs
        generation_inputs = inputs['input_ids']
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _logging_generate_kwargs(self, keys):
        if not hasattr(self, '_generate_kwargs'):
            self._generate_kwargs = None
        if self._generate_kwargs != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def save_predict(self, predict_results):
        if not self.is_world_process_zero():
            return

        preds, targets = predict_results.predictions, predict_results.label_ids
        origin_preds, origin_targets = preds, targets
        preds, targets = deepcopy(preds), deepcopy(targets)
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        preds[preds < 0] = self.tokenizer.pad_token_id
        targets[targets < 0] = self.tokenizer.pad_token_id
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        objs = []
        for p, t, pi, ti in zip(preds, targets, origin_preds.tolist(), origin_targets.tolist()):
            objs.append(dict(pred=p, target=t, pred_id=pi, target_id=ti))
        ret = {
            'metric': predict_results.metrics,
            'detail': objs,
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        json.dump(ret, open(os.path.join(self.args.output_dir, 'extra_predict.json'), 'w', encoding="utf-8"))
        self.log_metrics('predict', predict_results.metrics)  # noqa
        self.save_metrics('predict', predict_results.metrics)  # noqa
        return ret

    def plot_loss(self) -> None:
        training_args = self.args
        FIGURE_NAME = "trainer_state.png"
        import matplotlib.pyplot as plt
        data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
        train_steps, train_losses = [], []
        for i in range(len(data["log_history"]) - 1):
            train_steps.append(data["log_history"][i]["step"])
            train_losses.append(data["log_history"][i]["loss"])
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.title("training loss of {}".format(training_args.output_dir))
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
        print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


class Seq2SeqDataCollator(DataCollatorForSeq2Seq):
    def __init__(
            self,
            inference_mode: bool = False,
            **kwargs,
    ):
        self.inference_mode = inference_mode
        super().__init__(**kwargs)

    def __call__(self, features: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
        # evaluation set adopts left-padding while training set adopts right-padding
        if self.inference_mode:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            text_features = [{'input_ids': feature['input_ids'], 'labels': feature['labels']} for feature in features]
            ret = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side
            return ret
        input_ids, labels = [[torch.tensor(feature[key]) for feature in features] for key in ("input_ids", "labels")]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
        features = {"input_ids": input_ids, "labels": labels}
        return features


class Seq2Seq2DataCollatorWithImage(Seq2SeqDataCollator):
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)
        self.image_processor = preprocessor['image']
        self.zero_padding = torch.zeros(*self.image_processor.zero_padding_shape, dtype=torch.float)

    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        for feature in features:
            image = feature['image']
            if image is not None:
                images.append(self.image_processor(image))
            else:
                images.append(self.zero_padding)
        # B C=3 H=224 W=224
        images = torch.stack(images, dim=0)
        ret = dict(images=images)
        return ret

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        ret = super().__call__(features, return_tensors)
        image_outputs = self._image_process(features)
        ret.update(image_outputs)
        return ret
