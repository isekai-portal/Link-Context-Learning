IMAGENET_TEST = dict(
    type='ImageNet1kDataset',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_IMAGENET_TEST_VARIANT = dict(
    IMAGENET_SUB_TEST=dict(
        **IMAGENET_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/scripts/benchmark/imagenet_test.jsonl',
    ),
)
