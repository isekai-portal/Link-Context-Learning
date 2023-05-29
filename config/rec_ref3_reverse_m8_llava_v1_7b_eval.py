_base_ = ['_base_/dataset/rec_ref3_reverse.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/{{fileBasenameNoExtension}}',
)

data_args = dict(

    train=dict(
        caption_min_words=8,
        template_string='please generate an unambiguous description for the object <boxes> in the image.The generated caption should contain and only contain the query boxes.',
    ),
    validation=dict(
        caption_min_words=8,
        template_string='please generate an unambiguous description for the object <boxes> in the image.The generated caption should contain and only contain the query boxes.',
    ),
    test=dict(
        caption_min_words=8,
        template_string='please generate an unambiguous description for the object <boxes> in the image.The generated caption should contain and only contain the query boxes.',
    ),
)
