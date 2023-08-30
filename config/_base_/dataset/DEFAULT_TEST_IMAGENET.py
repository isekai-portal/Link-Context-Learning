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

IMAGENET1K2WAYCLEANISEKAI_TEST = dict(
    type='ImageNet1k2WayCleanISEKAIEval',
    filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/test10/isekai20_pos.json',
    image_folder=r'/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)