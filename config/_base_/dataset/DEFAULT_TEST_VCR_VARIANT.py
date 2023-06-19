VCR_VAL_COMMON_CFG = dict(
    type='VCRDataset',
    image_folder=r'sh41:s3://MultiModal/Monolith/academic/vcr/vcr1images',
    filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/vcr_val.jsonl',
)

VCR_TEST_COMMON_CFG = dict(
    type='VCRPredDataset',
    image_folder=r'sh41:s3://MultiModal/Monolith/academic/vcr/vcr1images',
    filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/vcr_test.jsonl',
)

DEFAULT_TEST_VCR_VARIANT = dict(
    VCR_val_qc_a=dict(**VCR_VAL_COMMON_CFG, version='qc-a', template_file=r"{{fileDirname}}/template/VQA.json", ),
    VCR_val_qc_rac=dict(**VCR_VAL_COMMON_CFG, version='qc-rac', template_file=r"{{fileDirname}}/template/VQA_BCoT.json", ),
    VCR_val_qac_r=dict(**VCR_VAL_COMMON_CFG, version='qac-r', template_file=r"{{fileDirname}}/template/VQA.json", ),
    VCR_val_qc_a_qc_r=dict(**VCR_VAL_COMMON_CFG, version='qc-a-qc-r', template_file=r"{{fileDirname}}/template/VQA.json", ),

    VCR_test_qc_a=dict(**VCR_TEST_COMMON_CFG, version='qc-a', template_file=r"{{fileDirname}}/template/VQA.json", ),
    VCR_test_qc_rac=dict(**VCR_TEST_COMMON_CFG, version='qc-rac', template_file=r"{{fileDirname}}/template/VQA_BCoT.json", ),
    VCR_test_qac_r=dict(**VCR_TEST_COMMON_CFG, version='qac-r', template_file=r"{{fileDirname}}/template/VQA.json", ),
    VCR_test_qc_a_qc_r=dict(**VCR_TEST_COMMON_CFG, version='qc-a-qc-r', template_file=r"{{fileDirname}}/template/VQA.json", ),
)

# ccfg = 'VCR_TEST_COMMON_CFG'
# splits = [
#     'val',
#     'test',
# ]
# versions = [
#     'qc-a', 'qc-ra', 'qc-rac',  # for evaluation: A
#     'qac-r', 'qc-a-qc-r',  # for evaluation: R
# ]
# for split in splits:
#     for v in versions:
#         name = f"VCR_{split}_{v.replace('-', '_')}"
#         filename = fr'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/vcr_{split}.jsonl'
#         print(f"{name}=dict(**{ccfg}, version='{v}', filename=r'{filename}'),")
