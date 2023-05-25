import re
import sys
import logging
from typing import List, Dict, Any, Tuple, Union

from ..utils.transform import norm_box_xyxy

from ..root import (
    FUNCTIONS,
    BaseTargetProcessFunc,
    BOXES_PLACEHOLDER,
    BOXES_PROCESSOR,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]


@FUNCTIONS.register_module()
class BoxFormatProcess(BaseTargetProcessFunc):
    def __call__(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], preprocessor: Dict[str, Any]
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        box_formatter = preprocessor['target']['boxes']

        # normalize target
        normalized_boxes = []
        if target is not None and 'boxes' in target:
            for box in target['boxes']:
                normalized_boxes.append(
                    norm_box_xyxy(box, w=target['width'], h=target['height'])
                )

        # convert bboxes_seq
        for sentence in raw_conv:
            words: str = sentence['value']
            boxes_seq: List[List[int]] = sentence.get('boxes_seq', None)
            if boxes_seq is None:
                continue
            # map box seq
            boxes_seq: List[Boxes] = map_box(normalized_boxes, boxes_seq)
            # reformat; replace <boxes> placeholder
            converted = box_formatter(words, boxes_seq)
            sentence['raw_value'] = sentence['value']
            sentence['value'] = converted

        return raw_conv, target


def map_box(boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_box(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    ret = []
    for boxes in boxes_seq:
        boxes_ret = []
        for box_index in boxes:
            boxes_ret.append(boxes_value[box_index])
        ret.append(boxes_ret)
    return ret


class BoxFormatter:
    def __init__(self, bboxes_token=BOXES_PLACEHOLDER, bboxes_token_pat=None):
        self.bboxes_token = bboxes_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        if bboxes_token_pat is None:
            bboxes_token_pat = bboxes_token
        self.bboxes_token_pat = re.compile(bboxes_token_pat)

    def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError


@BOXES_PROCESSOR.register_module()
class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        return "(" + box_str + ")"

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret


# FIXME: merge into load_pretrained
def smart_prepare_target_processor(
        model,  # multimodal llm
        preprocessor: Dict[str, Any],
        model_args,
        training_args,
):
    type_ = model_args.type
    if type_ in ['llava', 'otter']:
        return smart_prepare_target_processor_default(model, preprocessor, model_args, training_args)
    else:
        assert False


def smart_prepare_target_processor_default(
        model,  # multimodal llm
        preprocessor: Dict[str, Any],
        model_args,
        training_args,
):
    if not hasattr(model_args, 'target_processor'):
        return model, preprocessor

    target_processor = {}
    if 'boxes' in model_args['target_processor']:
        boxes_cfg = model_args['target_processor']['boxes']
        boxes_processor = BOXES_PROCESSOR.build(boxes_cfg)
        target_processor['boxes'] = boxes_processor
        # TODO: some boxes_formatter need adjust model/tokenizer
        #  luckily, our plain box formatter do not needed
        # if hasattr(boxes_processor, "post_process_model_tokenizer"):
        #     pass

    preprocessor['target'] = target_processor
    return model, preprocessor
