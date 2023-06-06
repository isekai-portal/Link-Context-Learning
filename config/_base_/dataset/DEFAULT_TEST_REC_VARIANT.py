REC_TEST_COMMON_CFG = dict(
    type='RECDataset',
    template_file=r'{{fileDirname}}/template/REC.json',
    max_dynamic_size=None,
)

DEFAULT_TEST_REC_VARIANT = dict(
    REC_REFCOCOG_UMD_TEST=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/REC_refcocog_umd_test.jsonl',
    ),
    REC_REFCOCOA_UNC_TEST=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/REC_refcoco+_unc_test.jsonl',
    ),
    REC_REFCOCO_UNC_TEST=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/REC_refcoco_unc_test.jsonl',
    ),
)
