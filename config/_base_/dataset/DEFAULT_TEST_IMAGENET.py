IMAGENET_TEST100 = dict(
    type='ImageNetTest100Eval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    # image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    image_folder=r'/mnt/lustre/share_data/taiyan/dataset/ImageNet-1K',
    template_file=r"{{fileDirname}}/template/ICL.json",
)
IMAGENET_TEST100_2WAY = dict(
    type='ImageNetTest100Eval2Way',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    # image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    image_folder=r'/mnt/lustre/share_data/taiyan/dataset/ImageNet-1K',
    template_file=r"{{fileDirname}}/template/ICL.json",
)