_base_ = ['_base_/dataset/DEFAULT_TRAIN_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    max_steps=599,
    save_steps=200,
    save_total_limit=2,
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)

data_args = dict(
    #
    train={{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_QC}},
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
