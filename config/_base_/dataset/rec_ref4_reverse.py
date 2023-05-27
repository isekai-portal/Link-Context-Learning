data_args = dict(

    train=dict(
        type='ReverseRECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/train.jsonl',
        template_string='please generate an unambiguous description for the object <boxes> in the image.'
    ),
    validation=dict(
        type='ReverseRECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/val.jsonl',
        template_string='please generate an unambiguous description for the object <boxes> in the image.'
    ),
    test=dict(
        type='ReverseRECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/test.jsonl',
        template_string='please generate an unambiguous description for the object <boxes> in the image.'
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
