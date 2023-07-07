import json
from glob import glob
import os.path
from pathlib import Path

from tqdm import tqdm

image_folder_root = r"D:\home\dataset\mscoco\images\train2014"
all_vg_test = [*list(glob('pointQA_*test.jsonl')),
               *list(glob('pointQA_*val.jsonl')),
               *list(glob('v7w_pointing_val.jsonl')),
               *list(glob('v7w_pointing_test.jsonl'))]


def all_test_images_ids():
    print(all_vg_test)
    all_test_images = set()
    for file in all_vg_test:
        for line in open(file):
            obj = json.loads(line)
            all_test_images.add(str(obj['genome_id']))
    print(len(all_test_images))
    json.dump(list(all_test_images), open('all_vg_test_images.json', 'w'))
    return all_test_images


def filter_train_item_vg(train_input_file, train_output_file, all_test_images_file, is_vg=False):
    all_test_images = json.load(open(all_test_images_file))
    print(all_test_images[:10])
    all_test_images = set(all_test_images)

    filtered_item = []
    with open(train_input_file, 'r', encoding='utf8') as f, open(train_output_file, 'w', encoding='utf8') as g:
        for line in tqdm(f):
            obj = json.loads(line)
            if is_vg:
                genome_id = Path(obj['img_path']).stem
            else:
                genome_id = obj['genome_id']
            genome_id = str(genome_id)
            if genome_id in all_test_images:
                filtered_item.append(obj)
                continue
            g.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"filtered: {len(filtered_item)}")
    json.dump(filtered_item, open(f"filtered_{train_output_file}", 'w'))
    return filtered_item


if __name__ == '__main__':
    all_test_images_ids()
