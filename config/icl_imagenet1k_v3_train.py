_base_ = ['_base_/dataset/DEFAULT_TRAIN_IMAGENET.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

# training_args = dict(
#     output_dir='/mnt/lustrenew/taiyan/models/shikra/{{fileBasenameNoExtension}}',
#     ceph_dir='ty-sdc:s3://ICL/checkpoint/taiyan/{{fileBasenameNoExtension}}/',
# )

dataset=dict(
    **_base_.IMAGENET1K_TRAIN,
    policy="policy_v3",
)

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {**dataset},
            {**dataset},
            {**dataset},
            {**dataset},
            {**dataset},
        ],
        probabilities=[0.2, 0.2, 0.2, 0.2, 0.2],
        seed=42,
        stopping_strategy='first_exhausted',
    ),
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
