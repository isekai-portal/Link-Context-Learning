import json

import jsonlines
from tqdm import tqdm

splits = ['val2014', 'train2014', 'test2015', 'test-dev2015']
imgdirs = ['val2014', 'train2014', 'test2015', 'test2015']
has_anns = [True, True, False, False]


def expand_image_idx(idx, tgt_len=12):
    idx = str(idx)
    assert len(idx) <= tgt_len
    ans = f"{'0' * (tgt_len - len(idx))}{idx}"
    return ans


for split, imgdir, has_ann in zip(splits, imgdirs, has_anns):
    filename = fr"D:\home\dataset\vqav2\v2_OpenEnded_mscoco_{split}_questions.json"
    output_file = f'v2_OpenEnded_mscoco_{split}_questions.jsonl'
    objs = json.load(open(filename))['questions']

    if has_ann:
        annotation_file = fr"D:\home\dataset\vqav2\v2_mscoco_{split}_annotations.json"
        anns = json.load(open(annotation_file))
        qid2ann = {}
        for item in anns['annotations']:
            qid2ann[item['question_id']] = item
        assert len(objs) == len(anns['annotations']) == len(qid2ann)

    with jsonlines.open(output_file, 'w') as g:
        for obj in tqdm(objs):
            image_full_idx = expand_image_idx(obj['image_id'])
            obj['image_path'] = f"{imgdir}/COCO_{imgdir}_{image_full_idx}.jpg"
            if has_ann:
                obj['annotation'] = qid2ann[obj['question_id']]
                assert obj['image_id'] == obj['annotation']['image_id']
            g.write(obj)
