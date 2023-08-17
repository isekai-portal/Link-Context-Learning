VQAv2_TEST_COMMON_CFG = dict(
    type='VQAv2Dataset',
    image_folder=r'zz1424:s3://publicdataset_49/VQAv2/unzip/',
    template_file=r"{{fileDirname}}/template/VQA_eval.json",
)

DEFAULT_TEST_VQAv2_VARIANT = dict(
    # VQAv2_val=dict(
    #     **VQAv2_TEST_COMMON_CFG,
    #     filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/v2_OpenEnded_mscoco_val2014_questions.jsonl',
    # ),
    # VQAv2_val_tiny=dict(
    #     **VQAv2_TEST_COMMON_CFG,
    #     filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/v2_OpenEnded_mscoco_val2014_questions.jsonl.tiny',
    # ),
    # VQAv2_testdev=dict(
    #     **VQAv2_TEST_COMMON_CFG,
    #     filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl',
    #     has_annotation=False,
    # ),
    VQAv2_test=dict(
        **VQAv2_TEST_COMMON_CFG,
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/v2_OpenEnded_mscoco_test2015_questions.jsonl',
        has_annotation=False,
    ),
)
