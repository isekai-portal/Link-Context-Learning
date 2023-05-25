from typing import Dict, Any, Tuple

from torch import nn

from .build_llava import load_pretrained_llava
from .build_otter import load_pretrained_otter

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if type_ == 'llava':
        return load_pretrained_llava(model_args, training_args)
    elif type_ == 'otter':
        return load_pretrained_otter(model_args, training_args)
    else:
        assert False
