import random
import jsonlines
from tqdm import tqdm

test100_pairs = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl'
test10_pairs = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test10_pairs.jsonl' 

classes = 'whippet, nematode, brain coral, ground beetle, pinwheel, printer, analog clock, african crocodile, green lizard, alligator lizard'
classes = classes.split(', ')

metas = []
with jsonlines.open(test100_pairs) as reader:
    for cls_meta in reader:
        if cls_meta["class_name"].lower() in classes:
            metas.append(cls_meta)


with jsonlines.open(test10_pairs, 'w') as writer:
    for meta in metas:
        writer.write(meta)

print('Finished')