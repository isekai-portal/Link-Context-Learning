import json
import os.path
from pathlib import Path

from tqdm import tqdm

from mllm.dataset.utils.io import read_img_general

image_folder_root = r"D:\home\dataset\mscoco\images\train2014"


def convert_ref3_jsonl():
    from datasets import load_dataset

    names = ["jxu124/refcoco", "jxu124/refcocog", "jxu124/refcocoplus"]
    dses = []

    all_test_images = set()
    for name in names:
        ds = load_dataset(name)
        dses.append(ds)
        for split, subds in ds.items():
            if split == 'train':
                continue
            _ = convert(name, split, subds)
            all_test_images = all_test_images.union(_)
    print(len(all_test_images))

    json.dump(list(all_test_images), open('all_test_images.json', 'w'))


def convert(name, split, subds):
    stem = Path(name).stem
    if stem in ['refcoco', 'refcocoplus'] and split == 'test':
        split = 'testA'
    save_name = f'{Path(name).stem}_{split}.jsonl'
    print(save_name)
    image_set = set()
    with open(save_name, 'w', encoding='utf8') as g:
        for item in tqdm(subds):
            image = read_img_general(os.path.join(image_folder_root, str(Path(item['image_path']).name)))
            bbox = eval(item['raw_anns'])['bbox']
            image_set.add(str(Path(item['image_path']).name))
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            assert bbox[0] < bbox[2] <= image.width and bbox[1] < bbox[3] <= image.height, f"{image} {bbox}"
            for sent in item['sentences']:
                obj = {
                    'img_path': str(Path(item['image_path']).name),
                    'expression': sent['sent'],
                    'bbox': bbox,
                    'dataset_name': name,
                    'height': image.height,
                    'width': image.width,
                    'image_id': item['image_id'],
                    'sent_id': sent['sent_id'],
                    'split': split,
                }
                g.write(json.dumps(obj) + '\n')
    return image_set


def filter_train_item(train_input_file, train_output_file, all_test_images_file, is_llava_data=False):
    all_test_images = json.load(open(all_test_images_file))
    all_test_images = set(all_test_images)

    filtered_item = []
    with open(train_input_file, 'r', encoding='utf8') as f, open(train_output_file, 'w', encoding='utf8') as g:
        for line in f:
            obj = json.loads(line)
            if is_llava_data:
                name = f"COCO_train2014_{obj['image']}"
            else:
                name = Path(obj['img_path']).name
            if name in all_test_images:
                filtered_item.append(obj)
                continue
            g.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"filtered: {len(filtered_item)}")
    json.dump(filtered_item, open(f"filtered_{train_output_file}", 'w'))
    return filtered_item


if __name__ == '__main__':
    convert_ref3_jsonl()
