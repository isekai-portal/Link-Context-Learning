import json
import os.path
import time
from typing import Dict, Any

from PIL import Image
from tqdm import tqdm

filename = r'/data/train.jsonl'
image_folder = r'D:\home\dataset\flickr30k\flickr30k-images'
template_string = r"caption the image"

from mllm.dataset.single_image_dataset import FlickrDataset

ds = FlickrDataset(
    filename=filename,
    image_folder=image_folder,
    template_string=template_string,
)

st = time.time()
for item in tqdm(ds):
    pass
print(time.time() - st)
