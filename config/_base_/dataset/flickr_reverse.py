data_args = dict(

    train=dict(
        type='FlickrBox2Caption',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k/train.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_string='please generate an unambiguous description for the object <boxes> in the image.',
        caption_with_box='question',
    ),
    validation=dict(
        type='FlickrBox2Caption',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k/val.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_string='please generate an unambiguous description for the object <boxes> in the image.',
        caption_with_box='question',
    ),
    test=dict(
        type='FlickrBox2Caption',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k/test.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_string='please generate an unambiguous description for the object <boxes> in the image.',
        caption_with_box='question',
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
