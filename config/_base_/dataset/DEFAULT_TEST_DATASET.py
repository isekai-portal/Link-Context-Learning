_base_ = [
    'DEFAULT_TEST_REC_VARIANT.py',
    'DEFAULT_TEST_FLICKR_VARIANT.py',
    'DEFAULT_TEST_GQA_VARIANT.py',
    'DEFAULT_TEST_CLEVR_VARIANT.py',
    'DEFAULT_TEST_GPTGEN_VARIANT.py',
    'DEFAULT_TEST_VCR_VARIANT.py',
    'DEFAULT_TEST_VQAv2_VARIANT.py',
    'DEFAULT_TEST_POINT_VARIANT.py',
    'DEFAULT_TEST_POPE_VARIANT.py',
    'DEFAULT_TEST_IMAGENET.py',
    'DEFAULT_TEST_IMAGENET_1K2WAY.py'
]

DEFAULT_TEST_DATASET = dict(
    **_base_.DEFAULT_TEST_REC_VARIANT,
    **_base_.DEFAULT_TEST_FLICKR_VARIANT,
    **_base_.DEFAULT_TEST_GQA_VARIANT,
    **_base_.DEFAULT_TEST_CLEVR_VARIANT,
    **_base_.DEFAULT_TEST_GPTGEN_VARIANT,
    **_base_.DEFAULT_TEST_VCR_VARIANT,
    **_base_.DEFAULT_TEST_VQAv2_VARIANT,
    **_base_.DEFAULT_TEST_POINT_VARIANT,
    **_base_.DEFAULT_TEST_POPE_VARIANT,
    **_base_.DEFAULT_IMAGENET_TEST_VARIANT,
    **_base_.DEFAULT_IMAGENET1K2WAY_TEST_VARIANT,
)
