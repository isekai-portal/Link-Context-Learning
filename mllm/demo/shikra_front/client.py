# client 端 api 调用案例---------------------------------
import os
import cv2
import re
import base64
import numpy as np
from typing import Union

import requests
from color_util import name_2_rgb, name_2_hex
from draw_bbox import draw_bounding_boxes


########################################
# helper
########################################

def cv2_to_base64(cv2_img):
    # 将图像编码为 base64 字符串
    _, encoded_img = cv2.imencode('.png', cv2_img)
    encoded_img_str = base64.b64encode(encoded_img).decode('utf-8')
    return encoded_img_str


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = int(x1 * w)
    x2 = int(x2 * w)
    y1 = int(y1 * h)
    y2 = int(y2 * h)
    box = x1, y1, x2, y2
    return box


# def expand2square(cv2_img, background_color=(255, 255, 255)):
#     # 获取图片的最大尺寸
#     h,w = cv2_img.shape[0], cv2_img.shape[1]
#     max_size = max(h,w)
#
#     # 使用 copyMakeBorder 函数将图片放到背景图中
#     x = (max_size - w) // 2
#     y = (max_size - h) // 2
#     pad_img = cv2.copyMakeBorder(cv2_img, y, y, x, x, cv2.BORDER_CONSTANT, value=background_color)
#
#     return pad_img


def squeeze_box(box, *, origin_size_w, origin_size_h):
    if origin_size_w == origin_size_h:
        return box
    if origin_size_w > origin_size_h:
        x1, y1, x2, y2 = box
        y1 -= (origin_size_w - origin_size_h) // 2
        y2 -= (origin_size_w - origin_size_h) // 2
        box = x1, y1, x2, y2
        return box
    assert origin_size_w < origin_size_h
    x1, y1, x2, y2 = box
    x1 -= (origin_size_h - origin_size_w) // 2
    x2 -= (origin_size_h - origin_size_w) // 2
    box = x1, y1, x2, y2
    return box


########################################
#
########################################

def query(image: Union[np.array, str], text: str, boxes_value: list, boxes_seq: list, server_url='http://127.0.0.1:12345/shikra'):
    if isinstance(image, str):
        image = cv2.imread(image)
    pload = {
        "img_base64": cv2_to_base64(image),
        "text": text,
        "boxes_value": boxes_value,
        "boxes_seq": boxes_seq,
    }
    resp = requests.post(server_url, json=pload)
    if resp.status_code != 200:
        raise ValueError(resp.reason)
    ret = resp.json()
    return ret


def postprocess(text, image):
    # print("text:", text)
    if image is None:
        return text, None

    # bbox & color
    colors = ['橙色', '钢蓝', '酸橙绿', '紫色', '猩红', '纯黄', '橄榄', '马鞍棕色', '青色']
    pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

    def extract_boxes(string):
        ret = []
        for bboxes_str in pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    extract_pred = extract_boxes(text)
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color_rgb = name_2_rgb(colors[idx % len(colors)])
        square_size = max(image.shape[1], image.shape[0])
        for box in boxes:
            denormed_box = de_norm_box_xyxy(box, w=square_size, h=square_size)
            squeezed_box = squeeze_box(denormed_box, origin_size_w=image.shape[1], origin_size_h=image.shape[0])
            boxes_to_draw.append(squeezed_box)
            color_to_draw.append(color_rgb)
    if not boxes_to_draw:
        return text, None

    # print(boxes_to_draw)
    # print(color_to_draw)

    res_img = draw_bounding_boxes(image, boxes_to_draw, color_to_draw)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color_rgb = name_2_rgb(colors[idx % len(colors)])
        color_hex = '#%02X%02X%02X' % color_rgb
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color_hex}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res_img


# 全图 vqa
def test1():
    image_path = os.path.join(os.path.dirname(__file__), 'examples/rec_bear.png')
    text = 'Can you point out a brown teddy bear with a blue bow in the image <image> and provide the coordinates of its location?'
    boxes_value = []
    boxes_seq = []

    response = query(image_path, text, boxes_value, boxes_seq, server_url)

    text_answer, image = postprocess(response['response'], image=cv2.imread(image_path))

    print(text_answer)
    if image is not None:
        cv2.imwrite("examples/test1_answer.png", image)


# bbox vqa
def test2():
    image_path = os.path.join(os.path.dirname(__file__), 'examples/man.jpg')
    text = "What is the person <boxes> scared of?"
    boxes_value = [[148, 99, 576, 497]]
    boxes_seq = [[0]]

    response = query(image_path, text, boxes_value, boxes_seq, server_url)

    text_answer, image = postprocess(response['response'], image=cv2.imread(image_path))

    print(text_answer)
    if image is not None:
        cv2.imwrite("examples/test2_answer.png", image)


if __name__ == '__main__':
    server_url = 'http://cluster-proxy.sh.sensetime.com:20358' + "/shikra"
    test1()
    test2()
