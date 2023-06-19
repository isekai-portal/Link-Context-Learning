_base_ = ['_base_/dataset/DEFAULT_TRAIN_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.flickr}},
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.reg}},
            {{_base_.DEFAULT_TRAIN_DATASET.gc}},
            {{_base_.DEFAULT_TRAIN_DATASET.caption}},
            dict(
                type='ConcatDataset',
                cfgs=[
                    {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_QBC}},
                    {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_RD_QBC}},
                ],
            )
        ],
        probabilities=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
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
