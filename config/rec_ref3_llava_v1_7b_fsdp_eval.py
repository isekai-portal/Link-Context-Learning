_base_ = ['_base_/dataset/rec_ref3.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp_eval.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/debug_exp_unify_mllm/{{fileBasenameNoExtension}}',
)
