GQA_TEST_COMMON_CFG = dict(
    type='GQADataset',
    filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/gqa_question_balanced_with_cot.jsonl',
    image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
    scene_graph_file=r"/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/gqa_scene_graph_data.jsonl",
    scene_graph_index=r"/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/gqa_scene_graph_index.json",
)

# use standard q-a mode
DEFAULT_TEST_GQA_VARIANT = dict(
    GQA_Q_A=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_Q_C=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_Q_BC=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_Q_S=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_Q_BS=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GQA_QB_A=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_QB_C=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QB_BC=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QB_S=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QB_BS=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GQA_QBP_A=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_QBP_C=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QBP_BC=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QBP_S=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QBP_BS=dict(**GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
)
