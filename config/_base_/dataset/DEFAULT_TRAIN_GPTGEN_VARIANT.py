GPTGEN_TRAIN_COMMON_CFG = dict(
    type='GPT4Gen',
    filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
)

DEFAULT_TRAIN_GPTGEN_VARIANT = dict(
    GPT4GEN_QA=dict(**GPTGEN_TRAIN_COMMON_CFG, version='a', template_file=r"{{fileDirname}}/template/VQA.json"),
    GPT4GEN_QC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='c', template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GPT4GEN_QBC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='bc', template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GPT4GEN_RD_QBC=dict(
        type=GPTGEN_TRAIN_COMMON_CFG['type'],
        image_folder=GPTGEN_TRAIN_COMMON_CFG['image_folder'],
        filename='/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
)
