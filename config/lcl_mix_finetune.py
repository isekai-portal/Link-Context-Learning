_base_ = ['_base_/dataset/mix_icl_prob_v13.py', '_base_/model/okapi_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    #output_dir='/mnt/lustre/fanweichen2/Research/MLLM/dummy_exp/{{fileBasenameNoExtension}}',
)