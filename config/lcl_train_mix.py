_base_ = ['_base_/dataset/mix_multitrain.py', '_base_/model/lcl_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    #output_dir='/mnt/lustre/fanweichen2/Research/MLLM/dummy_exp/{{fileBasenameNoExtension}}',
)