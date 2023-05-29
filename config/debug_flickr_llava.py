_base_ = ['_base_/dataset/flickr.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='./debug/{{fileBasenameNoExtension}}',
    overwrite_output_dir=True,
)

data_args = dict(

    train=dict(
        filename=r'D:\home\code\unify_mllm\data\train.jsonl',
        image_folder=r'D:\home\dataset\flickr30k\flickr30k-images',
        template_string="caption the image",
        template_file=None,
    ),
    validation=dict(
        filename=r'D:\home\code\unify_mllm\data\val.jsonl',
        image_folder=r'D:\home\dataset\flickr30k\flickr30k-images',
        template_string="caption the image",
        template_file=None,
    ),
    test=dict(
        filename=r'D:\home\code\unify_mllm\data\test.jsonl',
        image_folder=r'D:\home\dataset\flickr30k\flickr30k-images',
        template_string="caption the image",
        template_file=None,
    ),
)
