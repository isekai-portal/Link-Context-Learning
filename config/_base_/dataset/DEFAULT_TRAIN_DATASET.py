DEFAULT_TRAIN_DATASET = dict(
    flickr=dict(
        type='FlickrDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/CWB_flickr30k_train.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
    rec=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/REC_ref3_train.jsonl',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    reg=dict(
        type='REGDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/REC_ref3_train.jsonl',
        template_file=r'{{fileDirname}}/template/REG.json',
    ),
    gc=dict(
        type='GCDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/GC_genome196_train.jsonl',
        template_file=r'{{fileDirname}}/template/GC.json',
    ),
    caption=dict(
        type='CaptionDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/CAP_coco2014_train.jsonl',
        template_file=r'{{fileDirname}}/template/image_cap.json',
    ),
    instruct=dict(
        type='InstructDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/llava_instruct_150k.jsonl',
        image_folder=r'zz1424:s3://PublicDatalist/public_datalist_6_unzip/train2014',
    ),
)
