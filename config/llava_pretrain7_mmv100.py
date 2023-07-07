_base_ = ['_base_/dataset/mix_pretrain_prob.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
    tf32=False,
    bf16=False,
    fp16=True,
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
            {{_base_.DEFAULT_TRAIN_DATASET.GQA_Q_BC}},
            {{_base_.DEFAULT_TRAIN_DATASET.instruct}},
        ],
        probabilities=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7],
        seed=None,
        stopping_strategy='first_exhausted',
    ),
)