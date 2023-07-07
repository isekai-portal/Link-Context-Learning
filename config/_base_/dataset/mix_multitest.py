_base_ = ['DEFAULT_TEST_DATASET.py']

data_args = dict(
    #
    train=None,
    validation=None,
    test=None,
    mutlitest=dict(
        REC_REFCOCOG_UMD_TEST=dict(
            cfg={{_base_.DEFAULT_TEST_DATASET.REC_REFCOCOG_UMD_TEST}},
            compute_metric=dict(type='RECComputeMetrics'),
        ),
        REC_REFCOCOA_UNC_TEST=dict(
            cfg={{_base_.DEFAULT_TEST_DATASET.REC_REFCOCOA_UNC_TEST}},
            compute_metric=dict(type='RECComputeMetrics'),
        ),
        REC_REFCOCO_UNC_TEST=dict(
            cfg={{_base_.DEFAULT_TEST_DATASET.REC_REFCOCO_UNC_TEST}},
            compute_metric=dict(type='RECComputeMetrics'),
        ),
    ),

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
