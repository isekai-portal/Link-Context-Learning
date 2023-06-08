import json
import glob
from collections import defaultdict

images = set()
with open(f"./REC_ref3_train.jsonl", 'r') as f:
    for line in f:
        obj = json.loads(line)
        images.add(obj['img_path'])

all_test_images = set()
files = glob.glob('REC_refcoco*_test.jsonl')
for file in files:
    with open(file, 'r') as f:
        for line in open(file, 'r'):
            obj = json.loads(line)
            all_test_images.add(obj['img_path'])

test_images_not_in_train = all_test_images.difference(images)
print(len(test_images_not_in_train))

file2cnt = defaultdict(list)
with open("merged_REC_refcoco_test.jsonl", 'w') as g:
    for file in files:
        with open(file, 'r') as f:
            for line in open(file, 'r'):
                obj = json.loads(line)
                if obj['img_path'] in test_images_not_in_train:
                    obj['from'] = file
                    file2cnt[file].append(obj['img_path'])
                    g.write(json.dumps(obj))
                    g.write('\n')
