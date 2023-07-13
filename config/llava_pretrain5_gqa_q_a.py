_base_ = ['_base_/dataset/mix_pretrain_prob.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

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
            {{_base_.DEFAULT_TRAIN_DATASET.GQA_Q_A}},
        ],
        probabilities=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        seed=None,
        stopping_strategy='first_exhausted',
    ),
)