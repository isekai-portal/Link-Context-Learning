_base_ = ['_base_/dataset/mix_icl_imagenet1k.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustrenew/taiyan/models/shikra/{{fileBasenameNoExtension}}',
    # ceph_dir='ty-sdc:s3://MultiModal/checkpoint/taiyan/{{fileBasenameNoExtension}}/',
)