from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Any

from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainingArguments

DatasetDict = Dict[str, Dataset]
ComputeMetrics = Callable[[EvalPrediction], Dict]


# TODO: registry
def prepare_data(
        data_args,
        training_args: TrainingArguments,
        preprocessor: Dict[str, Any],
) -> Tuple[DatasetDict, ComputeMetrics]:
    if data_args.type == 'rec':
        return prepare_data_rec(data_args, training_args, preprocessor)
    else:
        assert False


def prepare_data_rec(
        data_args,
        training_args: TrainingArguments,
        preprocessor: Dict[str, Any],
) -> Tuple[DatasetDict, ComputeMetrics]:
    from .rec import RECDataset, RECComputeMetrics
    from .common import BOXFORMAT2FORMATTER, Expand2square
    from mllm.conversation import get_conv_template
    # get
    annotation_path = data_args.get('annotation_path', None)
    box_format_type = data_args.get('box_format_type', 'plain')
    box_format_kwargs = data_args.get('box_format_kwargs', {})
    expand2square = data_args.get('expand2square', None)
    conv_template = data_args.get('conv_template', 'vicuna_v1.1')
    tokenize_kwargs = data_args.get('tokenize_kwargs', {})
    template_string = data_args.get('template_string', None)
    template_file = data_args.get('template_file', None)
    max_dynamic_size = data_args.get('max_dynamic_size', None)
    if annotation_path is None:
        raise ValueError("please provide annotation_path to continue")
    # build
    box_formatter = BOXFORMAT2FORMATTER[box_format_type](**box_format_kwargs)
    dataset_func = partial(
        RECDataset,
        transform=Expand2square() if expand2square else None,
        preprocessor=preprocessor,
        training_args=training_args,
        box_formatter=box_formatter,
        conv_template=partial(get_conv_template, name=conv_template),
        tokenize_kwargs=tokenize_kwargs,
        template_string=template_string,
        template_file=template_file,
        max_dynamic_size=max_dynamic_size,
    )
    dataset = dict(
        train=dataset_func(data_file=Path(annotation_path) / 'train.jsonl', train_mode=True),
        validation=dataset_func(data_file=Path(annotation_path) / 'val.jsonl', train_mode=False),
        test=dataset_func(data_file=Path(annotation_path) / 'test.jsonl', train_mode=False),
    )
    compute_metrics = RECComputeMetrics(
        box_formatter=box_formatter,
        preprocessor=preprocessor,
    )
    return dataset, compute_metrics
