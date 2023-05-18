import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

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
        with torch.inference_mode():
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

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            # noinspection PyArgumentList
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.deepspeed and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.args.hf_deepspeed_config.dtype()})
            # vision input may contain float data and should be adjusted to match the dtypes
            # of the model while eval.
            elif (not self.is_in_train) and self.args.fp16_full_eval and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": torch.float16})
            elif (not self.is_in_train) and self.args.bf16_full_eval and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": torch.bfloat16})

            return data.to(**kwargs)
        return data

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
        # evaluation/inference adopts left-padding while training adopts right-padding
        if self.inference_mode:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            text_features = [{'input_ids': feature['input_ids'], 'labels': feature['labels']} for feature in features]
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side
            ret = {"input_ids": text_features['input_ids'], "labels": text_features['labels']}
        else:
            input_ids, labels = [[torch.as_tensor(feature[key]) for feature in features] for key in ("input_ids", "labels")]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
            ret = {"input_ids": input_ids, "labels": labels}
        # hack: this is only suit for llava
        ret['attention_mask'] = ret['input_ids'].ne(self.tokenizer.pad_token_id)
        return ret


class Seq2Seq2DataCollatorWithImage(Seq2SeqDataCollator):
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)
        self.image_processor = preprocessor['image']
        crop_size = self.image_processor.crop_size
        height, width = crop_size['height'], crop_size['width']
        self.zero_padding = torch.zeros(3, height, width, dtype=torch.float)

    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        for feature in features:
            image = feature['image']
            if image is not None:
                images.append(image)
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
