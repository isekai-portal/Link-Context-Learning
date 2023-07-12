from cmath import inf
import json
import os.path
import jsonlines

# Cluster 1424
# image root: [SenseData Ceph] s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/

# output_dir = '/mnt/cache/taiyan/test.jsonl'
# cls_id = '1'
# cls_name = 'test'
# cls_path = ['/path/to/image1', '/path/to/image2', '/path/to/image3']

# with jsonlines.open(output_dir, 'w') as writer:
#     writer.write([cls_id, cls_name, cls_path])

train_prefix = 'train/'
val_prefix = 'val/'

suffix = '.JPEG'
mappinp_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/devkit/data/map_clsloc.txt'

trainset_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/ImageSets/train_cls.txt'
valset_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/ImageSets/val.txt'

val_label_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
val_blacklist_path = 'ILSVRC2015_clsloc_validation_blacklist.txt'
# testset_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/ImageSets/test.txt'

output_dir = '/mnt/lustre/share_data/taiyan/dataset/imagenet1k/imagenet1k.jsonl'


def convert_imagenet1k_to_jsonl():
    folder2name = {}
    folder2id = {}
    id2name = {}
    with open(mappinp_path, 'r') as file:
        lines = file.readlines()
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            info = line.split(' ')
            folder = info[0]
            id = info[1]
            name = info[2]
            if folder not in folder2name.keys():
                folder2name[folder] = name
            if folder not in folder2id.keys():
                folder2id[folder] = id
            if id not in id2name.keys():
                id2name[id] = name

    objs = {}
    for id, name in id2name.items():
        objs[id] = [id, name, []]

    # train
    with open(trainset_path, 'r') as file:
        lines = file.readlines()
        train_labels = [
            folder2id[line.rstrip('\n').split(' ')[0].split('/')[0]]
            for line in lines
        ]
        train_metas = [
            train_prefix + line.rstrip('\n').split(' ')[0] + suffix
            for line in lines
        ]
        assert len(train_labels) == len(train_metas)
        for label, meta in zip(train_labels, train_metas):
            objs[label][2].append(meta)

    # val
    with open(val_label_path, 'r') as file:
        lines = file.readlines()
        val_labels = [line.rstrip('\n') for line in lines]
    with open(valset_path, 'r') as file:
        lines = file.readlines()
        val_metas = [
            val_prefix + line.rstrip('\n').split(' ')[0] + suffix
            for line in lines
        ]

    for label, meta in zip(val_labels, val_metas):
        objs[label][2].append(meta)

    with jsonlines.open(output_dir, 'w') as writer:
        for cls_id, cls_meta in objs.items():
            writer.write(cls_meta)


if __name__ == '__main__':
    convert_imagenet1k_to_jsonl()

    from tqdm import tqdm