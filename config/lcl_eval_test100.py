_base_ = ['_base_/dataset/DEFAULT_TEST_IMAGENET.py', '_base_/model/lcl_7b.py', '_base_/train/eval.py']

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

dataset=dict(
    **_base_.IMAGENET_TEST100_2WAY,
    sample_per_class=50,
    policy="policy_2way",
)

data_args = dict(
    train=None,
    validation=None,
    test=None,
    # ,
    multitest={"ImageNet1k_test100": {'cfg': dataset, 'compute_metric': dict(type='ImageNetTest100Metrics', filename=r'/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_pairs.jsonl')}},
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
