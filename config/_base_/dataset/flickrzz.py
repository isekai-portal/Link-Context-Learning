data_args = dict(

    train=dict(
        type='FlickrZz',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/train.jsonl',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),
    validation=dict(
        type='FlickrZz',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/val.jsonl',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),
    test=dict(
        type='FlickrZz',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/test.jsonl',
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
