_base_ = ['_base_/dataset/mix_pretrain_concat8.py', '_base_/model/llava_v1_13bo.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)