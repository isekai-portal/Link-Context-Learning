import re
import sys
import logging
from typing import List

Box = List[float]
Boxes = List[Box]
BoxesSeq = List[Boxes]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class BoxFormatter:
    def __init__(self, bboxes_token='<bboxes>', bboxes_token_pat=None):
        self.bboxes_token = bboxes_token
        # normally the bboxes_token_pat is the same as bboxes_token
        # if u not use some odd tokens
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


class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.pat = re.compile(r'\(\d(\.\d*)?(,\d(\.\d*)?){3}(;\d(\.\d*)?(,\d(\.\d*)?){3})*\)')

    def format_box(self, bboxes: Boxes) -> str:
        bbox_strs = []
        for bbox in bboxes:
            bbox_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in bbox]))
        box_str = ';'.join(bbox_strs)
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


BOXFORMAT2FORMATTER = {
    'plain': PlainBoxFormatter
}
