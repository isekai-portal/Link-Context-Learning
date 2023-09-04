from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn

from ..llava import LlavaLlamaForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint
from transformers import CLIPVisionModel
PREPROCESSOR = Dict[str, Any]

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_pretrained_llava(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    #model_args.model_name_or_path,
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    if 'qformer_config' in model_args:
        qformer_config = model_args.qformer_config
    else:
        qformer_config = None
    model_vision_dict = model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        qformer_config=qformer_config,
        dtype = dtype,
        device=training_args.device,
        freeze_mm_projector=model_args.freeze_mm_projector,
    )
    if 'qformer_config' in model_args:
        if model_args.qformer_config.load_model:
            print('loading qformer ckpt')
            missing_keys,unexpected_keys = load_sharded_checkpoint(model, model_args.model_name_or_path)
            print('missing: ',missing_keys)
            print('unexpected: ',unexpected_keys)

    #model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
    if isinstance(model.model.vision_tower, list):
        # HACK for quantization
        if model.model.vision_tower[0].device != torch.device('meta'):
            model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
        else:
            model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
            # model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip
    else:
        # HACK for quantization
        if model.model.vision_tower.device != torch.device('meta'):
            print("using meta device.")
            model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower, torch_dtype=dtype)
            # model.model.vision_tower.to(dtype=dtype, device=training_args.device)
        else:
            model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower, torch_dtype=dtype)  # not quantize clip
            print("not using meta device.")
            # model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip
        #print("is_deepspeed_zero3_enabled: ", is_deepspeed_zero3_enabled())
        print("vision tower's dtype & device: ", model.model.vision_tower.dtype, model.model.vision_tower.device)
        try:
            print(model.model.vision_tower.vision_model.embeddings.patch_embedding.weight.shape)
        except:
            print(model.model.vision_tower)

    vision_config = model_vision_dict['vision_config']

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter
    if model_args.freeze_mm_mlp_adapter:
        for p in model.model.mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # if len(params_no_grad) > 0:
    #     if training_args.fsdp is not None and len(training_args.fsdp) > 0:
    #         if len(params_no_grad) < 10:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
    #                                                                                                              params_no_grad))
    #         else:
    #             print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
    #                 len(params_no_grad), ', '.join(params_no_grad[:10])))
    #         print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
    #         print(
    #             "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

    #         from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    #         def patch_FSDP_use_orig_params(func):
    #             def wrap_func(*args, **kwargs):
    #                 use_orig_params = kwargs.pop('use_orig_params', True)
    #                 return func(*args, **kwargs, use_orig_params=True)
    #             return wrap_func

    #         FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            is_multimodal=model_args.is_multimodal,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
