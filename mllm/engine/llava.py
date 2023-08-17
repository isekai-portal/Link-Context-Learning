import os
import gc
import random
import json
import glob
import warnings
import shutil
from typing import Optional

import torch
import transformers
import numpy as np

from .base_engine import TrainerForMMLLM
from mllm.dataset.utils.io import delete_ceph, load_model_general, save_model_general, exists_ceph

from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.deepspeed import deepspeed_init
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, ShardedDDPOption
from transformers.trainer import (WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TRAINER_STATE_NAME,\
    OPTIMIZER_NAME, SCHEDULER_NAME,SCALER_NAME, PREFIX_CHECKPOINT_DIR, CONFIG_NAME, \
    unwrap_model, is_torch_tpu_available, is_sagemaker_mp_enabled, PreTrainedModel, logger)


# copy from transformers.modeling_utils, modified all "torch.load" to "load_model_general"
def load_sharded_checkpoint(model, local_dir, ceph_dir, strict=True):
    # Load the index
    index_file = os.path.join(local_dir, WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_file):
        raise ValueError(f"Can't find a checkpoint index ({WEIGHTS_INDEX_NAME}) in {local_dir}.")

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    for shard_file in shard_files:
        logger.info(f"Loading shard file in {os.path.join(ceph_dir, shard_file)}")
        state_dict = load_model_general(os.path.join(ceph_dir, shard_file), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is fred before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


class LLaVATrainer(TrainerForMMLLM):

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)
            
            # delete ceph model
            if self.args.ceph_dir:
                checkpoint_idx = checkpoint.split('/')[-1]
                ceph_checkpoint = os.path.join(self.args.ceph_dir, checkpoint_idx) + '/'
                delete_ceph(ceph_checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                save_model_general(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                save_model_general(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        #TODO：Modify the following process
        # super(LLaVATrainer, self)._save(output_dir, state_dict)
        logger = transformers.trainer.logger
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.args.ceph_dir:
            if(output_dir.split('/')[-1] == self.args.ceph_dir.split('/')[-2]):
                save_dir = self.args.ceph_dir
            else:
                save_dir = os.path.join(self.args.ceph_dir, output_dir.split('/')[-1]) + '/'
        else:
            save_dir = output_dir
        logger.info(f"Saving checkpoint to {save_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(save_dir, state_dict=state_dict, save_function=save_model_general)
            else:
                if state_dict is None:
                    state_dict = self.model.state_dict()
                save_model_general(state_dict, os.path.join(save_dir, transformers.trainer.WEIGHTS_NAME))
        else:
            self.model.save_pretrained(save_dir, state_dict=state_dict, save_function=save_model_general)

        # move (config.json, generation_config.json, pytorch_model.bin.index.json) to output_dir
        if save_dir != output_dir:
            if os.path.exists(save_dir):
                files = os.listdir(save_dir)
                for file in files:
                    source = os.path.join(save_dir, file)
                    target = os.path.join(output_dir, file)
                    shutil.move(source, target)
                shutil.rmtree(save_dir.split("//")[0])
            
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        save_model_general(self.args, os.path.join(output_dir, transformers.trainer.TRAINING_ARGS_NAME))

    # copy from transformers.trainer, modified all "torch.load" to "save_model_general"
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if self.args.ceph_dir:
            save_dir = os.path.join(self.args.ceph_dir, checkpoint_folder)
        else:
            save_dir = output_dir

        # the "output_dir" param in self.save_model() will be reload in self._save(), so don't worry
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            # taiyan: only save optimizer.pt into ceph, for consistency. 
            save_model_general(self.optimizer.state_dict(), os.path.join(save_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    # copy from transformers.trainer, modified all "torch.load" to "save_model_general"
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)) and not os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != transformers.__version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {transformers.__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if self.args.ceph_dir:
            checkpoint_idx = resume_from_checkpoint.split('/')[-1]
            model_dir = os.path.join(self.args.ceph_dir, checkpoint_idx) + '/'
        else:
            model_dir = resume_from_checkpoint
        logger.info(f"Actually, I'm loading model from {model_dir}.")

        weight_dir = os.path.join(model_dir, WEIGHTS_NAME)
        if os.path.isfile(weight_dir) or exists_ceph(weight_dir):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=model_dir, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = load_model_general(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu")
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = load_model_general(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu")
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(model, local_dir = resume_from_checkpoint, ceph_dir = model_dir, strict=is_sagemaker_mp_enabled())
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.deepspeed:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            return

        if self.args.ceph_dir:
            opt_dir = self.args.ceph_dir
        else:
            opt_dir = checkpoint

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (os.path.isfile(os.path.join(opt_dir, OPTIMIZER_NAME)) or exists_ceph(os.path.join(opt_dir, OPTIMIZER_NAME)))
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_tpu_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                optimizer_state = load_model_general(os.path.join(opt_dir, OPTIMIZER_NAME), map_location="cpu")
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = load_model_general(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(opt_dir, OPTIMIZER_NAME), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(opt_dir, OPTIMIZER_NAME), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(opt_dir, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    self.optimizer.load_state_dict(
                        load_model_general(os.path.join(opt_dir, OPTIMIZER_NAME), map_location=map_location)
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(load_model_general(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
                    self.scaler.load_state_dict(load_model_general(os.path.join(checkpoint, SCALER_NAME)))