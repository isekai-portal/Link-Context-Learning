_base_ = ['_base_/dataset/mix_pretrain_prob.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/cache/fanweichen2/Code/unify_mllm/tmp/{{fileBasenameNoExtension}}',
)