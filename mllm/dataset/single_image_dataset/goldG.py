# Mostly copy-paste from https://github.com/ashkamath/mdetr/blob/ea09acc44ca067072c4b143b726447ee7ff66f5f/datasets/mixed.py
import os
import os.path
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image
from torchvision.datasets.vision import VisionDataset

from ..root import DATASETS, BOXES_PLACEHOLDER
from ..utils import QuestionTemplateMixin, read_img_general, box_xywh_to_xyxy


def groupby_tokens_positive(target, tokens_positives: List[List[List[int]]]):
    from collections import defaultdict
    from itertools import chain
    ret = defaultdict(list)
    for idx, (tgt, tokens_positive) in enumerate(zip(target, tokens_positives)):
        tokens_positive = tuple(map(tuple, tokens_positive))
        _ = list(chain.from_iterable(tokens_positive))
        assert _ == sorted(_)
        ret[tokens_positive].append(idx)
    ret = dict(ret)
    return ret


def merge_tokens_positive(caption, tokens_positive: list):
    rets = []
    prev = [tokens_positive[0][0], tokens_positive[0][1]]
    for span in tokens_positive[1:]:
        inter = caption[prev[1]: span[0]]
        # inter is "" or "  "
        if (not inter) or inter.isspace():
            prev[1] = span[1]
        else:
            rets.append(prev)
            prev = [span[0], span[1]]
    rets.append(prev)
    return rets


@DATASETS.register_module()
class GoldGDataset(QuestionTemplateMixin, VisionDataset):
    """Coco-style dataset imported from TorchVision.
    It is modified to handle several image sources

    Args:
        root_coco (string): Path to the coco images
        root_vg (string): Path to the vg images
        ann_file (string): Path to json annotation file.
    """

    def __init__(
            self,
            *args,
            root_coco: str,
            root_vg: str,
            ann_file: str,
            **kwargs,
    ) -> None:
        super(GoldGDataset, self).__init__(*args, root=root_coco, **kwargs)
        from pycocotools.coco import COCO

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root_coco = root_coco
        self.root_vg = root_vg

    def __len__(self) -> int:
        return len(self.ids)

    def get_raw_item(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        dataset = img_info["data_source"]

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        img = read_img_general(os.path.join(cur_root, path))

        caption = img_info["caption"]
        return img, caption, target

    def __getitem__(self, index: int):
        # img
        image, caption_origin, target = self.get_raw_item(index)

        # caption
        pos2idxs = defaultdict(list)
        for idx, tgt in enumerate(target):
            phrase_ed_pos = tgt['tokens_positive'][-1][-1]
            pos2idxs[phrase_ed_pos].append(idx)

        sorted_pos = list(sorted(pos2idxs))

        caption_splited = list(caption_origin)
        boxes_seq = []
        for pos in sorted_pos:
            caption_splited[pos] = f"{BOXES_PLACEHOLDER}{caption_splited[pos]}"
            boxes_seq.append(pos2idxs[pos])
        caption_converted = "".join(caption_splited)
        boxes = [box_xywh_to_xyxy(tgt['bbox']) for tgt in target]

        # question
        question = self.get_template()

        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption_converted,
                    'boxes_seq': boxes_seq,
                }
            ]
        }
        return ret
