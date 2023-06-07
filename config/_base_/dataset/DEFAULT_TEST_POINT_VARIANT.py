# TODO: add test path
POINT_TEST_COMMON_CFG_LOCAL = dict(
    type='Point_QA_local',
    filename='',
    image_folder='',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

POINT_TEST_COMMON_CFG_TWICE = dict(
    type='Point_QA_twice',
    filename='',
    image_folder='',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

POINT_TEST_COMMON_CFG_V7W = dict(
    type='V7W_POINT',
    filename='',
    image_folder='',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

DEFAULT_TEST_GQA_VARIANT = dict(
    POINT_LOCAL_b=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='b'),
    POINT_LOCAL_p=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='p'),
    POINT_TWICE_oq_b=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-b'),
    POINT_TWICE_oq_p=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-p'),
    POINT_TWICE_sq_b=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-b'),
    POINT_TWICE_sq_p=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-p'),
    POINT_TWICE_gq_b=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-b'),
    POINT_TWICE_gq_p=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-p'),
    POINT_V7W_p=dict(**POINT_TEST_COMMON_CFG_V7W, version='p'),
    POINT_V7W_b=dict(**POINT_TEST_COMMON_CFG_V7W, version='b'),
)
