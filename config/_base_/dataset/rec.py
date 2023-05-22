data_args = dict(

    train=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/train.jsonl',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/rec_question_template.json',
        max_dynamic_size=1,
    ),
    validation=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/val.jsonl',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/rec_question_template.json',
        max_dynamic_size=1,
    ),
    test=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/test.jsonl',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref4-genome/rec_question_template.json',
        max_dynamic_size=1,
    ),

    compute_metric=dict(type='RECComputeMetrics'),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(),
    ),

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=128,
        num_beams=1,
    ),
)
