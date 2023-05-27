_base_ = ['_base_/dataset/flickr.py', '_base_/model/otter_9b.py', '_base_/train/otter_ds.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/{{fileBasenameNoExtension}}',
)
