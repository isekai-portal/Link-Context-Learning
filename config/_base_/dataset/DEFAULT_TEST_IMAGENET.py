IMAGENET1K_TEST = dict(
    type='ImageNet1kDatasetEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1K2WAY_YN_TEST = dict(
    type='ImageNet1k2WayYNEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1K2WAY_TEST = dict(
    type='ImageNet1k2WayEval',
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

IMAGENET1K2WAYISEKAI_TEST = dict(
    type='ImageNetISEKAI2wayEval',
    filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/test10/isekai20_pos.json',
    image_folder=r'/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1K2WAYCLEANISEKAI_TEST = dict(
    type='ImageNet1k2WayCleanISEKAIEval',
    filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/new/test10/isekai20_pos.json',
    image_folder=r'/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1KOPENNEGCLASS_TEST = dict(
    type='ImageNet1kOpenNegClassEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

IMAGENET1KTEST100_0SHOT = dict(
    type='Test100ZeroShot',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl',
    image_folder=r'ty1424:s3://production-public-imagenet/ImageNet/unzip/ILSVRC/Data/CLS-LOC/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

MINI_IMAGENET_5W1S = dict(
    type='MiniImageNetDatasetEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/lcl_val1000Episode_5_way_1_shot.json',
    image_folder=r'ty-sdc:s3://ICL/dataset/mini-imagenet/val/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

MINI_IMAGENET_5W5S = dict(
    type='MiniImageNetDatasetEval',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/mini-imagenet/lcl_val1000Episode_5_way_5_shot.json',
    image_folder=r'ty-sdc:s3://ICL/dataset/mini-imagenet/val/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)