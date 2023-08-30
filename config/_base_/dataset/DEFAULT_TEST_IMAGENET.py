IMAGENET1K_TEST = dict(
    type='ImageNet1kDatasetEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1K2WAYCLEAN_TEST = dict(
    type='ImageNet1k2WayCleanEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)