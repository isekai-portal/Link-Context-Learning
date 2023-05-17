from typing import Dict, Any, Tuple, Union

from PIL import Image


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def norm_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box


def norm_box_xyxy_expand2square(box, w, h):
    if w == h:
        return norm_box_xyxy(box, w, h)
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return norm_box_xyxy(box, w, w)
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return norm_box_xyxy(box, h, h)


class Expand2square:
    background_color = (255, 255, 255)

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Union[Image.Image, Tuple[Image.Image, Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image
        if 'bbox' in labels:
            bbox = norm_box_xyxy_expand2square(labels['bbox'], w=width, h=height)
            labels['bbox'] = bbox
        if 'bboxes' in labels:
            bboxes = [norm_box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['bboxes']]
            labels['bboxes'] = bboxes
        return processed_image, labels
