_base_ = ['_base_/dataset/DEFAULT_TRAIN_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

# training_args = dict(
#     output_dir='/mnt/lustrenew/taiyan/models/shikra/{{fileBasenameNoExtension}}',
#     ceph_dir='ty-sdc:s3://ICL/checkpoint/taiyan/{{fileBasenameNoExtension}}/',
# )

data_args = dict(
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_v9}},
            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_v10}},
            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_v11}},
        ],
        probabilities=[0.5,0.25,0.25],
        seed=None,
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

training_args = dict(
    num_train_epochs=25,
    save_strategy='no'
)
