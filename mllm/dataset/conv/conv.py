import sys
import copy
import logging
import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Tuple

import torch
from PIL import Image
from fastchat.conversation import SeparatorStyle
from torch.utils.data import Dataset
from transformers import TrainingArguments, LlamaTokenizer

from mllm.conversation import Conversation, get_conv_template

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def post_process(tokenizer, label):
    label = copy.deepcopy(label)
    label[label < 0] = tokenizer.pad_token_id
    return label


class ConvDatasetBase(Dataset):

    def __init__(
            self,
            *args,
            preprocessor,
            training_args: TrainingArguments = None,
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
            tokenize_kwargs: dict = None,
            train_mode: bool = True,
            conv_process_version: str = 'llava_v1',
            text_process_version: str = 'llava_v1',
            image_process_version: str = 'llava_v1',
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.preprocessor = preprocessor
        self.training_args = training_args
        self.conv_template = conv_template
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.text_process = Name2TextProcess[text_process_version]()
        self.image_process = Name2ImageProcess[image_process_version]()
        self.conv_process = Name2ConvProcess[conv_process_version]()
        self.train_mode = train_mode

    def __getitem__(self, index) -> Dict[str, Any]:
        raw_conv = self.get_conv_item(index)
        raw_conv = self.process_conv(raw_conv)
        conv, images = self.build_conv(raw_conv)
        text_dict = self.process_text(conv)
        image_dict = self.process_image(images)

        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        self._print_sample(ret_dict, raw_conv, conv)
        return ret_dict

    def get_conv_item(self, index) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def build_conv(self, source: List[Dict[str, Any]]) -> Tuple[Conversation, List[Image.Image]]:
        conv = self.conv_template()
        role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
        images = []
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        for sentence in source:
            role = role_map[sentence['from']]
            conv.append_message(role, sentence['value'])
            image = sentence.get('image', None)
            if image is not None:
                images.append(image)
        return conv, images

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.conv_process(raw_conv, self.preprocessor, self.conv_template)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        return self.text_process(conv, self.preprocessor, self.train_mode, **self.tokenize_kwargs)

    def process_image(self, images: List[Image.Image]) -> Dict[str, Any]:
        return self.image_process(images, self.preprocessor)

    def _print_sample(self, ret_dict, raw_conv, conv):
        if not hasattr(self, '_printed_sample'):
            self._printed_sample = True
            _mode = 'train' if self.train_mode else ' eval'
            post_processed_labels = post_process(self.preprocessor['text'], ret_dict['labels'])
            print(f"=================== {_mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
            if 'image' in ret_dict and ret_dict['image'] is not None:
                image = ret_dict['image']
                if isinstance(image, torch.Tensor):
                    print(f"            image: {image.shape}")
                elif isinstance(image, dict):
                    print(f"            image: {image.keys()}")
                elif isinstance(image, list) and len(image) > 0:
                    print(f"            image: {len(image)}, {type(image[0])}")
                else:
                    print(f"            image: {type(image)}")
            print("====================================================", flush=True)
            try:
                if self.training_args is not None:
                    _save_obj = {
                        'ret_dict': ret_dict,
                        'raw_conv': raw_conv,
                        'conv': conv.get_prompt(),
                    }
                    from pathlib import Path
                    output_dir = Path(self.training_args.output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    _mode = 'train' if self.train_mode else ' eval'
                    _local_rank = self.training_args.local_rank
                    _word_size = self.training_args.world_size
                    _file_path = str(output_dir / f'sample_check_{_mode}_{_local_rank}/{_word_size}.pt')
                    print(f'saving some sample to {_file_path} for check.')
                    torch.save(_save_obj, _file_path)
            except Exception as e:
                warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


# TODO registry
class LLavaConvProcessV1:
    def __call__(
            self,
            raw_conv: List[Dict[str, Any]],
            processor: Dict[str, Any],
            conv_template
    ) -> List[Dict[str, Any]]:
        multimodal_cfg = processor['multimodal_cfg']

        image_token_len = multimodal_cfg['image_token_len']
        is_multimodal = multimodal_cfg['is_multimodal']

        assert is_multimodal

        if multimodal_cfg['sep_image_conv_front']:
            assert DEFAULT_IMAGE_TOKEN in raw_conv[0]['value']
            raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            raw_conv[0]['value'] = DEFAULT_IMAGE_TOKEN + conv_template.sep + conv_template.roles[0] + ": " + raw_conv[0]['value']
        for sentence in raw_conv:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return raw_conv


Name2ConvProcess = {
    'llava_v1': LLavaConvProcessV1,
}


class LlavaTextProcessV1:
    def __call__(self, conv, processor, train_mode, **tokenizer_kwargs):
        tokenizer = processor['text']
        # the work of tokenize_conversation use the feature of tensor
        assert isinstance(tokenizer, LlamaTokenizer), "only work for LlamaTokenizer"
        _kwargs = {'return_tensors': 'pt'}
        _kwargs.update(tokenizer_kwargs)
        if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
            if train_mode:
                return self.tk_conv_colon_two_train(conv, tokenizer, **_kwargs)
            return self.tk_conv_colon_two_eval(conv, tokenizer, **_kwargs)
        else:
            raise ValueError(f"unrecognized conv_style: {conv.sep_style}.\n the conv is {conv}")

    def tk_conv_colon_two_train(self, conv, tokenizer, **kwargs):
        conversation = conv.get_prompt()
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0]
        target = copy.deepcopy(input_ids)
        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # <s> <space>
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                warnings.warn(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored):\n{conversation}")
        return dict(
            input_ids=input_ids,
            labels=target,
        )

    def tk_conv_colon_two_eval(self, conv, tokenizer, **kwargs):
        assert len(conv.messages) >= 2
        target = conv.messages[-1][-1]

        conv.messages[-1][-1] = ""
        conversation = conv.get_prompt()
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0]

        target = tokenizer([target, ], add_special_tokens=False, **kwargs).input_ids[0]
        target[target == tokenizer.pad_token_id] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            labels=target,
        )


Name2TextProcess = {
    'llava_v1': LlavaTextProcessV1,
}


class LlavaImageProcessorV1:
    def __call__(self, images, processor):
        image_processor = processor['image']

        crop_size = image_processor.crop_size
        height, width = crop_size['height'], crop_size['width']

        assert len(images) in [0, 1]
        if len(images) == 1:
            image = images[0]
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = torch.zeros(3, height, width)
        return {'image': image}


Name2ImageProcess = {
    'llava_v1': LlavaImageProcessorV1,
}
