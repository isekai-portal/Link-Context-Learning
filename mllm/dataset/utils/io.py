import io
import os
import sys
import time
import logging

import cv2
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def read_img_general(img_path):
    if "s3://" in img_path:
        cv_img = read_img_ceph(img_path)
        # noinspection PyUnresolvedReferences
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        return Image.open(img_path).convert('RGB')

def load_model_general(model_path, map_location='cpu'):
    if "s3://" in model_path:
        ckpt = load_model_ceph(model_path,map_location=map_location)
        return ckpt
    else:
        return torch.load(model_path, map_location=map_location)

def save_model_general(state_dict, save_path):
    if "s3://" in save_path:
        ckpt = save_model_ceph(state_dict, save_path)
        return ckpt
    else:
        return torch.save(state_dict, save_path)

client = None


def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f"Please check image at {img_path}"
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    # noinspection PyUnresolvedReferences
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def load_model_ceph(model_path, map_location ="cpu"):
    init_ceph_client_if_needed()
    value = client.get(model_path)
    model_buffer = io.BytesIO(value)
    ckpt = torch.load(model_buffer, map_location=map_location)
    return ckpt

def save_model_ceph(state_dict, save_path):
    init_ceph_client_if_needed()
    with io.BytesIO() as f:
        torch.save(state_dict, f)
        value = f.getvalue()
        client.put(save_path, value)

def exists_ceph(path):
    init_ceph_client_if_needed()
    client.contains(path)

def listdir_ceph(path):
    init_ceph_client_if_needed()
    return client.list(path)

def delete_ceph(path):
    init_ceph_client_if_needed()
    if path.endswith('/'):
        contents = listdir_ceph(path)
        for content in contents:
            delete_ceph(os.path.join(path, content))
    else:
        client.delete(path)

def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client(enable_mc=True)
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")