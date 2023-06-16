_base_ = ['_base_/dataset/mix_pretrain_concat8.py', '_base_/model/llava_v1_13bo.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    num_train_epochs=3,
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=1024),
    ),
)
