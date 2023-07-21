IMAGENET1K2WAY_TEST = dict(
    type='ImageNet1k2WayDataset',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_IMAGENET1K2WAY_TEST_VARIANT = dict(
    IMAGENET1K2WAY_SUB_TEST=dict(
        **IMAGENET1K2WAY_TEST,
        filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/imagenet1k_pairs.jsonl',
    ),
)
