_base_ = ['_base_/dataset/DEFAULT_TEST_DATASET.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

training_args = dict(
    tf32=False,
    bf16=False,
    fp16=True,
    fsdp=False,

    do_train=False,
    do_eval=False,
    do_predict=True,
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
)

data_args = dict(
    #
    train=None,
    validation=None,
    test={{_base_.DEFAULT_TEST_DATASET.POINT_LOCAL_p_test}},

    # compute_metric
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
