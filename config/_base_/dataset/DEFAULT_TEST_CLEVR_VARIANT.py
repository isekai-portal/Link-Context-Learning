CLEVR_TEST_COMMON_CFG = dict(
    type='ClevrDataset',
    filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/CLEVR_val_questions_with_ans.jsonl',
    image_folder=r'',
    scene_graph_file=None,
)

DEFAULT_TEST_CLEVR_VARIANT = dict(
    CLEVR_A_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA.json",
    ),
    CLEVR_S_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA_CoT.json",
    ),
    CLEVR_BS_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA_PCoT.json",
    ),
)
