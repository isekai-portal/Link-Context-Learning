IMAGENET1K_TRAIN = dict(
    type='ImageNet1kDatasetTrain',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/train900_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_TRAIN_IMAGENET1K_VARIANT = dict(
    imagenet1k_train=dict(**IMAGENET1K_TRAIN),
)