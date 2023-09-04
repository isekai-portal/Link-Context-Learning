BUSINESS_VQA_TEST = dict(
    type='BusinessVQADataset_MIX',
    template_file=r"{{fileDirname}}/template/ICL.json",
)
ISEKAI_VQA_TEST = dict(
    type='ISEKAIVQADataset',
    template_file=r"{{fileDirname}}/template/ICL.json",
)
# PRE IDcard_jubao_det/
# PRE Logo2ktest/
# PRE RuiAn/
# PRE banner_test/
# PRE bloody_test/
# PRE certificate_classify_test/
# PRE crendiential_test/
# PRE fire_test/
# PRE flag_det/
# PRE flag_tvlogo_precision/
# PRE knife_det/
# PRE knife_test/
# PRE language/
# PRE lihan/
# PRE logo_det_adela/
# PRE qrcode_det_test/
# PRE text_det_v2/
# PRE video_classify_v2/
# PRE weapon_classify_test/
# PRE yellow_ribbon/

# DEFAULT_ISEKAI_NEW=dict(
#     EQUIMANOID = dict(
#         **ISEKAI_VQA_TEST,
#         filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/all.txt',
#         filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/positive.txt',
#         filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/negative.txt',
#         image_folder_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         image_folder_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         image_folder=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         label='equimanoid',
#         label_negative='horse',
#     ),
# )

# DEFAULT_ISEKAI=dict(
#     EQUIMANOID = dict(
#         **BUSINESS_VQA_TEST,
#         filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/all.txt',
#         filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/positive.txt',
#         filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/files/hosre_man/negative.txt',
#         image_folder_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         image_folder_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         image_folder=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/academic/data/',
#         label='equimanoid',
#         label_negative='horse',
#     ),
# )
DEFAULT_BUSINESS_DETECTION= dict(
    BANNER_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/banner/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/banner/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/banner/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/banner_test/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/banner_test/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/banner_test/',
        label='banner',
    ),
    CHAIR_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/chair/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/chair/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/chair/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/chair/images/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/chair/images/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/chair/images/',
        label='chair',
    ),
    DOG_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/dog/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/dog/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/dog/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/dog/images/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/dog/images/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/dog/images/',
        label='dog',
    ),
    IDCARD_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/id_card/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/id_card/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/id_card/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        label='id_card',
    ),
    KNIFE_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/knife/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/knife/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/knife/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/knife_det/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/knife_det/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/knife_det/',
        label='knife',
    ),
    PLASTICBAG_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/plastic_bag/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/plastic_bag/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/plastic_bag/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/plastic_bag/images/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/plastic_bag/images/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/Smart_City/det/plastic_bag/images/',
        label='plastic_bag',
    ),
    QRCODE_DET=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/qr_code/top_100.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/qr_code/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy_glip/qr_code/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
        label='qr_code',
    ),
)
DEFAULT_BUSINESS_EASY= dict(
    CREDENTIAL_GT=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/crendiential_file/all.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/crendiential_file/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/crendiential_file/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/crendiential_test/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/crendiential_test/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/crendiential_test/',
        label='credential_file',
    ),
    BANNER_GT=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/BANNER/all.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/BANNER/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/BANNER/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/banner_test/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/flag_det/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/banner_test/',
        label='banner',
    ),
    IDCARD_GT=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/ID_CARD/all.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/ID_CARD/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/ID_CARD/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        label='ID_card',
    ),
    QRCODE_GT=dict(
        **BUSINESS_VQA_TEST,
        filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/QR_CODE/all.txt',
        filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/QR_CODE/positive.txt',
        filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/QR_CODE/negative.txt',
        image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
        image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/IDcard_jubao_det/',
        image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
        label='qr_code',
    ),
)

