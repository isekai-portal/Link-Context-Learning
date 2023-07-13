from typing import Dict, Any, Tuple

import transformers
from torch import nn
from transformers import LlamaConfig

from ..otter import OtterForConditionalGeneration, OtterConfig

PREPROCESSOR = Dict[str, Any]


def load_pretrained_otter(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    if hasattr(model_args, 'build_small_model') and model_args.build_small_model:        
        config = OtterConfig.from_pretrained(model_args.model_name_or_path)
        config.text_config.update(dict(num_attention_heads=2, num_hidden_layers=6))
        otter = OtterForConditionalGeneration(config, model_args.tokenizer_name_or_path)
    else:
        otter = OtterForConditionalGeneration.from_pretrained(model_args.model_name_or_path, model_args.tokenizer_name_or_path)
    tokenizer = otter.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()

    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=model_args.conv_processor,
    )

    return otter, preprocessor
