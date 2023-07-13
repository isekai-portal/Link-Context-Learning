_base_ = ['_base_/dataset/DEFAULT_TEST_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/{{fileBasenameNoExtension}}',

    do_train=False,
    do_eval=False,
    do_predict=False,
    do_multi_predict=True,

    fp16=True,
    fp16_full_eval=True,
    bf16=False,
    bf16_full_eval=False,
    per_device_eval_batch_size=8,
)

model_args = dict(
    model_name_or_path=None,
)

data_args = dict(
    train=None,
    validation=None,
    test=None,

    multitest=dict(
        dataset1=dict(
            cfg=dict(
                type='PureVQADataset',
                filename='/mnt/lustre/share_data/chenkeqin/mllm_data/eval_data/dummy_pure_vqa_dataset1.jsonl',
                image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images',
                template_string='<image> <question>',
            ),
            compute_metric=None,
        ),
        dataset2=dict(
            cfg=dict(
                type='PureVQADataset',
                filename='/mnt/lustre/share_data/chenkeqin/mllm_data/eval_data/dummy_pure_vqa_dataset2.jsonl',
                image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014',
                template_string='<image> <question>',
            ),
            compute_metric=None,
        ),
        # 手写一个VQA_BCoT的模板
        dataset3=dict(
            cfg=dict(
                type='PureVQADataset',
                filename='/mnt/lustre/share_data/chenkeqin/mllm_data/eval_data/dummy_pure_vqa_dataset2.jsonl',
                image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014',
                template_string='<image> <question> Make sure to offer a clear explanation and indicate object locations with [xmin,ymin,xmax,ymax].',
            ),
            compute_metric=None,
        ),
        # 用VQA_BCoT中的模板
        dataset4=dict(
            cfg=dict(
                type='PureVQADataset',
                filename='/mnt/lustre/share_data/chenkeqin/mllm_data/eval_data/dummy_pure_vqa_dataset2.jsonl',
                image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014',
                template_file=r'{{fileDirname}}/_base_/dataset/template/VQA_BCoT.json',
            ),
            compute_metric=None,
        ),
    ),

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
