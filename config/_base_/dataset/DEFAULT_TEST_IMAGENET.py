IMAGENET_TEST = dict(
    type='ICLEvalDataset',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_IMAGENET_TEST_VARIANT = dict(
    IMAGENET_SUB_TEST=dict(
        **IMAGENET_TEST,
        filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
        sample_per_class=50,
    ),
)
