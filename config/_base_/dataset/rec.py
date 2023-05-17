data_args = dict(
    type='rec',
    annotation_path='/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome',

    # image
    expand2square=True,

    # box
    box_format_type='plain',
    box_format_kwargs={},

    # conv
    conv_template='vicuna_v1.1',
    tokenize_kwargs={},

    # question template
    template_string=None,
    template_file=None,
    max_dynamic_size=None,

    # padding collator kwargs
    padding=True,
    max_length=1024,

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
        synced_gpus=True,
    )
)
