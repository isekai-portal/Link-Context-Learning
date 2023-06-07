_base_ = ['_base_/dataset/DEFAULT_TEST_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/{{fileBasenameNoExtension}}',

    do_train=False,
    do_eval=True,
    do_predict=True,

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
    validation={{_base_.DEFAULT_TEST_DATASET.GQA_Q_BC_BALANCED}},
    test={{_base_.DEFAULT_TEST_DATASET.GQA_Q_A_BALANCED}},

    compute_metric=dict(type='GQAComputeMetrics'),

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=128,
        num_beams=1,
    ),
)
