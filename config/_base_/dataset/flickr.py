data_args = dict(

    train=dict(
        type='FlickrDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos/train.jsonl',
        annotation_dir=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),
    validation=dict(
        type='FlickrDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos/val.jsonl',
        annotation_dir=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),
    test=dict(
        type='FlickrDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos/test.jsonl',
        annotation_dir=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_annos',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=256,
        num_beams=1,
    ),
)
