V3DET_TRAIN_COMMON_CFG = dict(
    type='V3DetDataset',
    filename=r'/mnt/lustre/share_data/zhangzhao2/VG/v3det/v3det_2023_v1_train_neig_expired_fix.json',
    image_folder=r'sdc:s3://mm_data/v3det/',
    template_file=r"{{fileDirname}}/template/DOD.json",
)

DEFAULT_TRAIN_V3DET_VARIANT = dict(
    v3det_dod_train=dict(**V3DET_TRAIN_COMMON_CFG),
)
