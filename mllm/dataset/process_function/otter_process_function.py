import re
from typing import Dict, Any, List

import torch
from PIL import Image

from ..root import (
    FUNCTIONS,
    IMAGE_PLACEHOLDER,
    BaseImageProcessFunc,
    BaseConvProcessFunc,
    BaseTextProcessFunc,
)
from ...conversation import Conversation

IGNORE_INDEX = -100
MEDIA_TOKEN = "<image>"
ENDOFCHUNK_TOKEN = "<|endofchunk|>"
ANSWER_TOKEN = "<answer>"


@FUNCTIONS.register_module()
class OtterConvProcess(BaseConvProcessFunc):
    def __call__(self, raw_conv: List[Dict[str, Any]], preprocessor: Dict[str, Any], conv_template: Conversation) -> List[Dict[str, Any]]:
        conv_processor_cfg = preprocessor['conv']
        max_src_length = conv_processor_cfg.get('max_src_length', None)
        max_tgt_length = conv_processor_cfg.get('max_tgt_length', None)
        image_token_at_begin = conv_processor_cfg.get('image_token_at_begin', False)

        for sentence in raw_conv:
            if sentence['from'] == 'human':
                item = pre_question(sentence['value'], max_src_length)
                if image_token_at_begin and IMAGE_PLACEHOLDER in item:
                    item = item.replace(IMAGE_PLACEHOLDER, "")
                    item = f"{IMAGE_PLACEHOLDER} {item}"
                sentence['value'] = item
            if sentence['from'] == 'gpt':
                sentence['value'] = pre_question(sentence['value'], max_tgt_length)
        return raw_conv


@FUNCTIONS.register_module()
class OtterTextProcess(BaseTextProcessFunc):
    def __call__(self, conv: Conversation, preprocessor: Dict[str, Any], mode: str, **tokenize_kwargs) -> Dict[str, Any]:
        tokenizer = preprocessor['text']

        _kwargs = {'return_tensors': 'pt'}
        _kwargs.update(tokenize_kwargs)

        if mode in ['train']:
            return self.tk_conv_colon_two_train(conv, tokenizer, **_kwargs)
        return self.tk_conv_colon_two_eval(conv, tokenizer, **_kwargs)

    def tk_conv_colon_two_train(self, conv, tokenizer, **kwargs):
        assert len(conv.messages) == 2, f'not support multi-round conversation training by now. but get message len: {len(conv.messages)}'
        media_token_id = tokenizer.get_vocab()[MEDIA_TOKEN]
        answer_token_id = tokenizer.get_vocab()[ANSWER_TOKEN]

        conversation = conv.get_prompt()
        src_text = tokenizer(conversation, add_special_tokens=False, **kwargs)
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)

        bos_item = torch.LongTensor([tokenizer.bos_token_id])
        eos_item = torch.LongTensor([tokenizer.eos_token_id])
        bos_mask = torch.LongTensor([1])
        eos_mask = torch.LongTensor([1])

        input_ids = torch.cat([bos_item, src_item, eos_item])
        attention_mask = torch.cat([bos_mask, src_item_mask, eos_mask])

        label_ids = input_ids.clone()
        assert answer_token_id in label_ids
        label_ids[label_ids == tokenizer.pad_token_id] = IGNORE_INDEX
        label_ids[label_ids == tokenizer.bos_token_id] = IGNORE_INDEX
        label_ids[label_ids == media_token_id] = IGNORE_INDEX
        label_idx = 0
        while label_idx < len(label_ids) and label_ids[label_idx] != answer_token_id:
            label_ids[label_idx] = IGNORE_INDEX
            label_idx += 1
        label_ids[label_ids == answer_token_id] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label_ids,
        )

    def tk_conv_colon_two_eval(self, conv, tokenizer, **kwargs):
        assert len(conv.messages) >= 2
        media_token_id = tokenizer.get_vocab()[MEDIA_TOKEN]

        target = conv.messages[-1][-1]
        conv.messages[-1][-1] = ""
        conversation = conv.get_prompt()
        src_text = tokenizer(conversation, add_special_tokens=False, **kwargs)
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)

        bos_item = torch.LongTensor([tokenizer.bos_token_id])
        bos_mask = torch.LongTensor([1])

        input_ids = torch.cat([bos_item, src_item])
        attention_mask = torch.cat([bos_mask, src_item_mask])

        target = tokenizer([target, ], add_special_tokens=False, **kwargs).input_ids[0]
        target[target == tokenizer.pad_token_id] = IGNORE_INDEX
        target[target == tokenizer.bos_token_id] = IGNORE_INDEX
        target[target == media_token_id] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target,
        )


@FUNCTIONS.register_module()
class OtterImageProcess(BaseImageProcessFunc):
    def __call__(self, image: Image.Image, preprocessor: Dict[str, Any]) -> Dict[str, Any]:
        image_processor = preprocessor['image']

        if image is not None:
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            if hasattr(image_processor, 'crop_size'):
                crop_size = image_processor.crop_size
                height, width = crop_size['height'], crop_size['width']
            else:
                raise ValueError("got empty image. and don't know how to pad")
            image = torch.zeros(3, height, width)
        # N T C H W
        image = image.unsqueeze(1).unsqueeze(1)
        return {'image': image}


def pre_question(question, max_ques_words=None):
    question = (
        question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
    )

    question = re.sub(
        r"\s{2,}",
        " ",
        question,
    )
    question = question.rstrip("\n")
    question = question.strip(" ")

    # truncate question
    if max_ques_words is not None:
        question_words = question.split(" ")
        if len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

    return question


def pre_answer(answer, max_ans_words=None):
    answer = re.sub(
        r"\s{2,}",
        " ",
        answer,
    )
    answer = answer.rstrip("\n")
    answer = answer.strip(" ")

    # truncate question
    if max_ans_words is not None:
        return_answer = ""
        answers = answer.split(".")

        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(" ")) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(" ")
            return_answer = " ".join(answer_words[:max_ans_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."
    else:
        return_answer = answer
    return return_answer
