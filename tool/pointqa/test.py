from itertools import cycle
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
from mllm.dataset.root import DATASETS, BOXES_PLACEHOLDER, IMAGE_PLACEHOLDER
from mllm.dataset.utils import MInstrDataset
from mllm.dataset.utils.flickr30k_entities_utils import (
    flatten_annotation,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
)
from PIL import Image, ImageDraw
class Point_QA_twice(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(),template_string='')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['file_path']
        obj_question = item['obj_question']
        super_question = item['super_question']
        general_question = item['general_question']
        answer = item['answer']
        image = self.get_image(img_path)
        bbox = item['bbox']
        point = item['point']

        ret = {
            'image': image,
            'target': {'boxes': bbox,'point':point},
            'question':{'obj_question':obj_question,'super_question':super_question,'general_question':general_question},
            'answer':answer
        }
        return ret
class Point_QA_local(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(),template_string='')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['file_path']
        question = item['question']
        answer = item['answer']
        image = self.get_image(img_path)
        bbox = item['bbox']
        point = item['point']

        ret = {
            'image': image,
            'target': {'boxes': bbox,'point':point},
            'question':question,
            'answer':answer
        }
        return ret
class V7W_POINT(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(),template_string='')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['file_path']
        question = item['question']
        answer = item['answer']
        image = self.get_image(img_path)
        bbox = item['candidates']
        point = item['point']

        ret = {
            'image': image,
            'target': {'boxes': bbox,'point':point},
            'question':question,
            'answer':answer
        }
        return ret
def draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='green')

def draw_point(image, point):
    draw = ImageDraw.Draw(image)
    x, y = point
    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill='red', outline='red')

if __name__ == "__main__":
    #point_qa = Point_QA_local(filename='./data/pointQA_local_train.jsonl',image_folder='s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/')
    #point_qa = Point_QA_twice(filename='./data/pointQA_twice_train.jsonl',image_folder='s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/')
    point_qa = V7W_POINT(filename='./data/v7w_pointing_train.jsonl',image_folder='sh41:s3://MultiModal/Monolith/academic/v7w/data')
    for i in range(0,30):
        det = point_qa[i]
        image = det['image']
        bbox = det['target']['boxes']
        point = det['target']['point']
        question = det['question']
        answer = det['answer']

        # 在图像中标出矩形框和点
        for candidates in bbox:
            draw_bbox(image, candidates)
        draw_point(image, point)

        # 在图像上绘制问题和答案
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # 自定义字体和大小
        text_position = (10, 10)  # 文本位置

        draw.text(text_position, f'Question: {question}', font=font, fill='white')
        draw.text((text_position[0], text_position[1] + 20), f'Answer: {answer}', font=font, fill='white')

        # 保存图像为 PNG 格式
        image.save(('output_'+str(i)+'.png'), 'PNG')