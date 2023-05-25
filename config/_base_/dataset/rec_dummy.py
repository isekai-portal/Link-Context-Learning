_base_ = ['rec.py']

data_args = dict(

    train=dict(
        type='RECDataset',
        filename=r'D:\home\code\unify_mllm\data\dummy_conv_ann.jsonl',
        template_string='<expr>',
        template_file=None,
    ),
    validation=dict(
        type='RECDataset',
        filename=r'D:\home\code\unify_mllm\data\dummy_conv_ann.jsonl',
        template_string='<expr>',
        template_file=None,
    ),
    test=dict(
        type='RECDataset',
        filename=r'D:\home\code\unify_mllm\data\dummy_conv_ann.jsonl',
        template_string='<expr>',
        template_file=None,
    ),
)
