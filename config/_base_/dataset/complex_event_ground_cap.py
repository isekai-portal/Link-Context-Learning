import glob

data_args = dict(

    train=None,
    validation=None,
    test=dict(
        type='ComplexEventGroundCap',
        filename=glob.glob('/mnt/lustre/chenkeqin/share_data/zhangzhao2/Monolith/business/anno/**/*train*.jsonl', recursive=True),
        image_folder=r'/mnt/lustre/chenkeqin/share_data/zhangzhao2/Monolith/business/data',
        template_string='please generate an unambiguous description for the object <boxes> in the image.',
    ),

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=512,
        num_beams=1,
    ),
)
