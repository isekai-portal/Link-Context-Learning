import pytest


@pytest.fixture(scope='session')
def cfg_dir():
    import platform
    from pathlib import Path
    from argparse import Namespace

    llava_7b_tk_path = rf"{Path(__file__).parent / 'llava_7b_tk'}"
    llama_7b_hf_path = rf"{Path(__file__).parent / 'llama_7b_hf'}"

    if 'LAPTOP' in platform.node():
        coco_images_dir = r'D:\home\dataset\mscoco\images'
    elif '142-4' in platform.node():
        coco_images_dir = r'/mnt/lustre/share_data/chenkeqin/resources/coco'
    elif '198-6' in platform.node():
        coco_images_dir = r'/mnt/lustre/share_data/chenkeqin/resources/coco'
    else:
        coco_images_dir = r''

    return Namespace(
        LLAVA_7B_TK_PATH=llava_7b_tk_path,
        LLAMA_7B_HF_PATH=llama_7b_hf_path,
        COCO_IMAGES_DIR=coco_images_dir,
    )