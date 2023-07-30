_base_ = ['_base_/dataset/DEFAULT_TEST_ISEKAI.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

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

dataset1=dict(
    **_base_.ISEKAI,
    sample_per_class=0,
    policy="policy_v6",
)
dataset2=dict(
    **_base_.ISEKAI,
    sample_per_class=0,
    policy="mini_v6",
)

data_args = dict(
    train=None,
    validation=None,
    test=None,
    multitest={
        "isekai_horse_man_true": {'cfg': dataset1, 'compute_metric': dict(type='ICLComputeMetrics')},
        "isekai_equimanoid_fake": {'cfg': dataset2, 'compute_metric': dict(type='ICLComputeMetrics')}
    },
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
