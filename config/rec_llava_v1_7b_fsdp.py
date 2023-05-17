_base_ = ['_base_/dataset/rec.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/exp_rec/{{fileBasenameNoExtension}}',
)
