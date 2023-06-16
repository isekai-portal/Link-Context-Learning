_base_ = ['_base_/dataset/mix_pretrain_concat8.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)

model_args = dict(
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/llava_pretrain3+concat+recvg+e1',
)
