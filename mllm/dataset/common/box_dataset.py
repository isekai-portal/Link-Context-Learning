import sys
import json
import random
import logging
from typing import Dict, Any, List

from .box_formatter import BoxesSeq, BoxFormatter, PlainBoxFormatter
from ..conv import ConvDatasetBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class QuestionTemplateMixin:
    def __init__(self, *args, template_string=None, template_file=None, max_dynamic_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_string = template_string
        self.template_file = template_file
        self.max_dynamic_size = max_dynamic_size
        if template_string is None and template_file is None:
            raise ValueError("assign either template_string or template_file")
        if template_string is not None and template_file is not None:
            raise ValueError(f"assign both template_string and template_file:\nstring:{template_string}\nfile:{template_file}")
        if template_string is not None:
            self.templates = [self.template_string]
        else:
            assert template_file is not None
            self.templates = json.load(open(template_file, 'r', encoding='utf8'))
        if self.max_dynamic_size is not None:
            self.templates = self.templates[: self.max_dynamic_size]

    def get_template(self):
        return random.choice(self.templates)

    def template_nums(self):
        return len(self.templates)


class BoxDatasetBase(ConvDatasetBase):
    def __init__(self,
                 *args,
                 box_formatter: BoxFormatter = PlainBoxFormatter(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter = box_formatter

    def get_conv_item(self, index) -> List[Dict[str, Any]]:
        raw_conv: List[Dict[str, Any]] = self.get_raw_conv_with_box(index)
        raw_conv: List[Dict[str, Any]] = self.convert_box(raw_conv)
        return raw_conv

    def get_raw_conv_with_box(self, index) -> List[Dict[str, Any]]:
        """
        {
            from: human
            value: balabala<boxes>balabala<boxes>
            image: image
            boxes_seq: [[[x_min, y_min, x_max, y_max][x_min, y_min, x_max, y_max]]
                        [[x_min, y_min, x_max, y_max]]]
        },{
            from: gpt
            value: balabala<boxes>
            image: None
            boxes_seq: [[[x_min, y_min, x_max, y_max]]]
        }
        """
        raise NotImplementedError

    def convert_box(self, source: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for sentence in source:
            words: str = sentence['value']
            bboxes_seq: BoxesSeq = sentence.get('bboxes_seq', None)
            if bboxes_seq is None:
                continue
            converted = self.box_formatter(words, bboxes_seq)
            sentence['raw_value'] = sentence['value']
            sentence['value'] = converted
        return source
