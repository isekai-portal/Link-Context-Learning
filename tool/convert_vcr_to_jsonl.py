import json
import os.path

import jsonlines


def convert_vcr_to_jsonl():
    splits = ['val', 'test', 'train']

    image_folder = f""
    for split in splits:
        filename = fr"D:\home\dataset\vcr1annots\{split}.jsonl"
        output_file = f'vcr_{split}.jsonl'

        with jsonlines.open(filename) as reader, jsonlines.open(output_file, 'w') as writer:
            for obj in reader:
                metadata_fn = obj['metadata_fn']
                meta = json.load(open(os.path.join(image_folder, metadata_fn)))
                obj['boxes'] = meta['boxes']
                obj['width'] = meta['width']
                obj['height'] = meta['height']
                writer.write(obj)


if __name__ == '__main__':
    convert_vcr_to_jsonl()

    from tqdm import tqdm